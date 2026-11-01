# Future experiments — RQ0 (not run)

Entries: idea, what prompted it, why it is not run now. Writing an idea here is the complete response to it.

1. **Does the cat student also shift toward other animals (dolphin, owl, ...)?** Prompted by the E1e naming test (`results_RQ0.md` is silent on this): on cat data the dolphin prompt explained the data best (74% of bags), cat only 5%, which suggests a generic animal-persona shift. RQ0 scored only the substring "cat". The stored `eval_{base,control,cat}.jsonl` already hold all 10,000 completions per model, so a post-hoc count of "dolphin" (and the other nine animals) per model needs no new model run. Not run now: it is a new metric outside the frozen plan, and a post-hoc scan of the answers could not be pre-registered. If done, label it exploratory, report all ten animals (not only dolphin), and compare with the base and control models.
2. **Direct "do you like dolphins?" probe on the students.** Same motivation. Needs new sampling from the saved adapters (`data/qwen/cat_adapter`, `control_adapter`) on the HPC. Not run now: new GPU job outside the frozen plan.
