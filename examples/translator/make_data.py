"""Build the gozer / tt-smi command-translator data set.

Every command below uses only flags that appear in the real `--help` output of
gozer and tt-smi on this box (checked 2026-11-23). Each intent has six English
phrasings; four go to training and two only to the test set, and slot values
(lease ids, tickets, reasons, scripts) come from disjoint train/test pools, so
the test set measures generalisation and not recall.

Output: train.jsonl, test.jsonl with {"text": "Request: ...\\nCommand: ...<eos>"}
for training and {"request", "command"} for test (kept in test.jsonl).
"""
import json
import random
from pathlib import Path

random.seed(20261123)
HERE = Path(__file__).resolve().parent
EOS = "<|endoftext|>"  # SmolLM2 / GPT-2 style end marker, also the stop signal at eval

# --- slot pools (disjoint between train and test) -----------------------
WHO = {"train": ["claude:keymaster", "claude:tt-finetune", "claude:optimize", "alice", "bench-bot"],
       "test": ["claude:orchard", "claude:tt-lang", "bob", "ci-runner"]}
REASON = {"train": ["run the unit tests", "profile the decoder", "benchmark training steps", "smoke test the install",
                    "collect a trace", "validate the adapter"],
          "test": ["check the new firmware", "debug a hang", "measure decode speed", "reproduce the crash"]}
CMD = {"train": ["pytest tests/test_decoder.py", "python train.py", "python bench.py --steps 50", "tt-smi -s",
                 "python examples/quickstart.py"],
       "test": ["pytest tests/test_attention.py", "python serve.py", "python eval.py --split dev",
                "python examples/resume_training.py"]}
HEX = "0123456789abcdef"
def rid(n): return "".join(random.choice(HEX) for _ in range(n))
POOL_LEASE = {"train": [rid(6) for _ in range(12)], "test": [rid(6) for _ in range(6)]}
POOL_TICKET = {"train": [rid(4) for _ in range(12)], "test": [rid(4) for _ in range(6)]}
FILES = {"train": ["snap.json", "before.json", "state.json", "run1.json"], "test": ["after.json", "board.json", "nightly.json"]}
DUR = {"train": ["30m", "45m", "2h", "10m"], "test": ["1h", "90m", "20m"]}
CHIPS = {"train": ["1", "2", "3", "4"], "test": ["1", "2", "4"]}
NAMES = {"train": ["one", "two", "three", "four"], "test": ["one", "two", "four"]}  # spelled-out chip counts

# --- intents: (six phrasings, command builder). {slots} are filled per split ---
def intents():
    return [
        (["show who holds the chips", "who has the chips right now", "what is the current lease state",
          "list the leases and who is waiting", "give me the gozer status", "is anyone using the boards"],
         lambda s: "gozer status"),
        (["show the chip and board topology", "what boards and chips does this box have", "print the topology",
          "list the chips grouped by board", "how are the chips laid out", "describe the hardware layout"],
         lambda s: "gozer topology"),
        (["list the waiting requests", "show the queue", "who is queued for chips", "print pending lease requests",
          "what is waiting for a lease", "show me the wait queue"],
         lambda s: "gozer queue"),
        (["show past leases", "what leases ran before", "print the lease history", "who held chips earlier",
          "list finished leases", "show the gozer history"],
         lambda s: "gozer history"),
        (["what is being served and where", "list the running services", "show which models are being served",
          "which ports are leased for serving", "print the served models", "show active services"],
         lambda s: "gozer services"),
        (["take a lease on {n} chips as {who} for {reason}", "acquire {n} chips, I am {who}, reason: {reason}",
          "I need {n} chips for {reason}, my name is {who}", "lease {n} chips for {who} to {reason}",
          "get me {n} chips as {who}, because I want to {reason}", "reserve {n} chips under {who}: {reason}"],
         lambda s: f'gozer acquire --chips {s["n"]} --who {s["who"]} --reason "{s["reason"]}"'),
        (["run {cmd} on {n} chips as {who} to {reason}", "lease {n} chips as {who} and run {cmd} to {reason}",
          "execute {cmd} under a {n} chip lease, who is {who}, to {reason}", "I am {who}, run {cmd} with {n} chips, reason {reason}",
          "use {n} chips as {who} to run {cmd}, to {reason}", "run {cmd} on {n} chips for {who}: {reason}"],
         lambda s: f'gozer run --chips {s["n"]} --who {s["who"]} --reason "{s["reason"]}" -- {s["cmd"]}'),
        (["wait for ticket {ticket}", "block until ticket {ticket} is granted", "hold until ticket {ticket} is ready",
          "wait on queue ticket {ticket}", "pause until {ticket} gets chips", "wait until my ticket {ticket} comes up"],
         lambda s: f'gozer wait {s["ticket"]}'),
        (["release lease {lease}", "give back lease {lease}", "I am done with lease {lease}, free it",
          "hand back the chips of lease {lease}", "end lease {lease}", "return lease {lease}"],
         lambda s: f'gozer release {s["lease"]}'),
        (["cancel ticket {ticket}", "drop queue ticket {ticket}", "remove ticket {ticket} from the queue",
          "I no longer need ticket {ticket}, cancel it", "withdraw ticket {ticket}", "scrap queued request {ticket}"],
         lambda s: f'gozer cancel {s["ticket"]}'),
        (["extend lease {lease} by {dur}", "renew lease {lease} for {dur}", "keep lease {lease} alive, expect {dur} more",
          "heartbeat lease {lease} with an expected {dur}", "lease {lease} needs {dur} longer", "refresh lease {lease}, {dur} expected"],
         lambda s: f'gozer renew {s["lease"]} --expect {s["dur"]}'),
        (["print the exports for lease {lease}", "show the environment lines for lease {lease}", "give me export lines for {lease}",
          "which env variables does lease {lease} set", "env for lease {lease}", "show shell exports of lease {lease}"],
         lambda s: f'gozer env {s["lease"]}'),
        (["reset the chips of lease {lease} in place", "reset lease {lease} but keep it held", "re-initialise the chips on lease {lease}",
          "do an in-place reset of lease {lease}", "reset the boards for lease {lease} without releasing", "restart the chips under lease {lease}"],
         lambda s: f'gozer reset {s["lease"]}'),
        (["take a snapshot of the devices", "dump the tt-smi snapshot", "print device telemetry as json",
          "show a snapshot of the hardware", "get the current device snapshot", "snapshot all tt devices to stdout"],
         lambda s: "tt-smi -s"),
        (["list the boards on the host", "show the boards", "which boards are installed", "list the tenstorrent devices",
          "print board types and bus ids", "what boards does tt-smi see"],
         lambda s: "tt-smi -ls"),
        (["save a snapshot to {file}", "write the device snapshot into {file}", "dump tt-smi info to the file {file}",
          "store the snapshot as {file}", "snapshot the devices into {file}", "put the hardware snapshot in {file}"],
         lambda s: f'tt-smi -f {s["file"]}'),
        (["open tt-smi in compact mode without the network", "run tt-smi compact and offline", "tt-smi, compact view, no network access",
          "show the compact tt-smi without versions box", "start tt-smi offline in compact mode", "compact offline tt-smi please"],
         lambda s: "tt-smi -c --offline"),
    ]

def fill(template, split):
    n = random.choice(CHIPS[split])
    slots = dict(n=n, who=random.choice(WHO[split]), reason=random.choice(REASON[split]),
                 cmd=random.choice(CMD[split]), ticket=random.choice(POOL_TICKET[split]),
                 lease=random.choice(POOL_LEASE[split]), file=random.choice(FILES[split]),
                 dur=random.choice(DUR[split]))
    return template.format(**slots), slots

def build(split, phrasing_idx, per_phrase):
    rows = []
    for phrasings, make in intents():
        for i in phrasing_idx:
            for _ in range(per_phrase):
                req, slots = fill(phrasings[i], split)
                rows.append({"request": req, "command": make(slots)})
    random.shuffle(rows)
    return rows

train = build("train", [0, 1, 2, 3], 11)   # 16 intents x 4 phrasings x 11 = 704
test = build("test", [4, 5], 6)            # 16 intents x 2 phrasings x 6 = 192
with open(HERE / "train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps({"text": f'Request: {r["request"]}\nCommand: {r["command"]}{EOS}'}) + "\n")
with open(HERE / "test.jsonl", "w") as f:
    for r in test:
        f.write(json.dumps(r) + "\n")
print(len(train), "train rows,", len(test), "test rows")
# Leak check: no test request or (request, command) pair appears in training text.
train_text = " ".join(json.dumps(r) for r in train)
leaks = [r for r in test if r["request"] in train_text]
print("test requests found verbatim in train:", len(leaks))
