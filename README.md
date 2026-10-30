# Demos

Small, self-contained scripts that each demonstrate one idea. Every demo lives in its own folder with its own README, code, and saved output, so you can read the results without running anything.

| Demo | What it shows |
|---|---|
| [bacteria-cog](bacteria-cog/) | Cognition without consciousness: six cognitive processes in a foraging agent |

## Adding a demo

1. Copy [`_template/`](_template/) to a new folder named after the demo.
2. Write the demo and fill in its README.
3. Add a row to the table above.

Each demo keeps its own `requirements.txt`, since demos need different libraries.

## Tools

[`tools/lab`](tools/lab) starts long jobs on a lab machine and shows what is running. `lab start` checks that a job is actually running before it returns, and `lab status` shows every job as running, done, FAILED or LOST, plus any problem lines in its log. Run `tools/lab` with no arguments for usage.
