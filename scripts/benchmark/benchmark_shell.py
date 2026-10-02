"""Compare fresh and reused PowerShell using harmless commands, without an API.

Run with the same Python interpreter used by the installed Jarv launcher.
Startup is measured separately; subsequent samples alternate runner order.
"""

import argparse
import json
import platform
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jarv.shell import ShellState, execute_command


def measure(state, command, persistent):
    started = time.perf_counter()
    result = execute_command(command, shell_state=state, persistent_shell=persistent)
    elapsed = (time.perf_counter() - started) * 1000
    if result.exit_code != 0:
        raise RuntimeError(result.full_model_output())
    return round(elapsed, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps', type=int, default=7)
    args = parser.parse_args()
    if platform.system() != 'Windows':
        parser.error('This benchmark measures the Windows PowerShell runner.')
    if args.reps < 1:
        parser.error('--reps must be positive')
    python = "'" + sys.executable.replace("'", "''") + "'"
    fresh, reused = ShellState.initial(), ShellState.initial()
    report = {'python': sys.executable, 'repetitions': args.reps, 'cases': []}
    try:
        report['first_reused_command_ms'] = measure(reused, "'ok'", True)
        if reused._worker is None or reused._worker.closed:
            raise RuntimeError('The persistent worker fell back; comparison would be invalid.')
        worker = reused._worker
        for label, command in [('print ok', "'ok'"), ('git version', 'git --version'),
                               ('Python no-op', f'& {python} -c "pass"')]:
            samples = {'fresh': [], 'reused': []}
            for index in range(args.reps):
                for name in (('fresh', 'reused') if index % 2 else ('reused', 'fresh')):
                    samples[name].append(measure(reused if name == 'reused' else fresh,
                                                 command, name == 'reused'))
            if reused._worker is not worker or worker.closed:
                raise RuntimeError('Worker was not reused throughout the benchmark.')
            medians = {name: round(statistics.median(values), 2) for name, values in samples.items()}
            report['cases'].append({'command': label, 'samples_ms': samples, 'median_ms': medians,
                                    'saved_ms': round(medians['fresh'] - medians['reused'], 2)})
    finally:
        fresh.close()
        reused.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
