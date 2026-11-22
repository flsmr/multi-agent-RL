"""Build the report from recorded experiment results; optionally print it to PDF."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads((ROOT / path).read_text())


def table(headers, rows):
    head = ''.join(f'<th>{escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def figure(name, caption):
    return f'<figure><img src="assets/{name}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def duration(seconds):
    minutes, remainder = divmod(round(seconds), 60)
    return f'{minutes}m {remainder:02d}s'


def score(summary):
    return f"{summary['mean_score']:.3f} ± {summary['seed_mean_std']:.3f}"


def build_report():
    overview = read_json('runs/experiment_summary.json')
    main = overview['main_experiment']
    teacher = read_json(main['teacher_run'] + '/config.json')
    student = read_json(main['student_run'] + '/config.json')
    collection = read_json(main['collection_run'] + '/config.json')
    dataset = read_json(main['dataset'] + '/manifest.json')
    evaluations = {name: read_json(path + '/summary.json') for name, path in main['evaluations'].items()}
    if any(s['status'] != 'complete' for s in evaluations.values()) or teacher['target_episode'] is None:
        raise ValueError('Finish training and all evaluations before publishing the report.')
    milestones = {item['threshold']: item for item in teacher['score_milestones']}
    half, one = milestones[.5], milestones[1.]
    stoch, deterministic, student_eval = (evaluations[name] for name in
                                         ('teacher_stochastic', 'teacher_deterministic', 'student_deterministic'))
    with (ROOT / main['student_run'] / 'losses.csv').open() as stream:
        losses = list(csv.DictReader(stream))
    selected = next(row for row in losses if int(row['epoch']) == student['best_epoch'])
    history = [
        ('Initial PPO, seed 0', 'teacher_ppo_seed0'),
        ('Initial PPO, seed 2', 'teacher_ppo_seed2'),
        ('Adjusted single environment', 'teacher_ppo_longroll_seed2'),
        ('Parallel, long rollout', 'teacher_ppo_parallel8_longroll_seed2'),
        ('Parallel, short rollout; stop 0.5', 'teacher_ppo_parallel8_roll512_seed2'),
        ('Main: parallel, short rollout; stop 1', Path(main['teacher_run']).name),
    ]
    history_rows = []
    for label, name in history:
        c = read_json(f'runs/{name}/config.json')
        history_rows.append((label, f"{c.get('num_envs',1)} × {c['rollout_steps']:,}",
                             f"{c['hyperparameters']['update_epochs']} / {c['hyperparameters']['gamma']}",
                             f"{c['best_rolling100']:.3f}",
                             f"{c['solved_environment_steps']:,}" if c.get('solved_environment_steps') else 'Not solved'))
    old_student = read_json('runs/evaluation_student_deterministic/summary.json')
    old_teacher = read_json('runs/evaluation_teacher_deterministic/summary.json')
    long_stoch = read_json('runs/teacher_ppo_parallel8_longroll_seed2/evaluation_stochastic_seeds/summary.json')
    long_det = read_json('runs/teacher_ppo_parallel8_longroll_seed2/evaluation_deterministic_seeds/summary.json')
    short_stoch = read_json('runs/teacher_ppo_parallel8_roll512_seed2/evaluation_stochastic_seeds/summary.json')
    short_det = read_json('runs/teacher_ppo_parallel8_roll512_seed2/evaluation_deterministic_seeds/summary.json')
    teacher_higher = deterministic['mean_score'] > student_eval['mean_score']
    comparison = ('The deterministic teacher and distilled student have effectively equal mean scores in this evaluation.'
                  if abs(deterministic['mean_score'] - student_eval['mean_score']) < .01 else
                  'The deterministic teacher scores higher than its distilled student in this run.' if teacher_higher else
                  'The distilled student scores higher than the deterministic teacher in this run.')
    stage_sum = sum(c['elapsed_seconds'] for c in (teacher, collection, student))
    pages = []
    pages.append(f'''
    <p class="eyebrow">Udacity · Collaboration and Competition · 21 November 2026</p>
    <h1>PPO Teacher and<br>Local Student for Tennis</h1>
    <p class="lead">A shared teacher learns with both players’ observations. A smaller student then learns its actions from local observations alone.</p>
    <div class="callout"><strong>Main experiment:</strong> eight parallel environments, 512 steps per environment,
    training continued to a 100-episode mean of 1.0. The teacher first passes 0.5 at <strong>{half['environment_steps']:,} environment steps</strong>
    and 1.0 at <strong>{one['environment_steps']:,} steps</strong>. The final distilled student scores
    <strong>{score(student_eval)}</strong> across five fresh evaluation seeds.</div>
    {table(['Frozen policy', 'Inputs/player', 'Mean ± SD', 'Seeds ≥ 0.5'], [
        ('Teacher, stochastic', 48, score(stoch), f"{stoch['passed_seeds']}/5"),
        ('Teacher, deterministic', 48, score(deterministic), f"{deterministic['passed_seeds']}/5"),
        ('Student, deterministic', 24, score(student_eval), f"{student_eval['passed_seeds']}/5")])}
    <p class="small">Each seed mean covers 100 complete games. SD is the sample standard deviation of the five seed means.
    These are evaluations of one policy, not independent training repetitions.</p>
    <h2>1. Problem and scoring</h2>
    <p>Two simulated players cooperate to keep a ball in play. A successful return earns +0.1;
    dropping the ball or sending it out of bounds incurs −0.01. Each player receives eight local position/velocity values,
    stacked over three frames into 24 inputs, and controls horizontal motion and jumping with two continuous actions in [−1, 1].</p>
    <p>An episode score is the larger of the two players’ total undiscounted returns. The official solve criterion is a mean
    of at least 0.5 over 100 consecutive completed episodes. The extended training target of 1.0 is an additional experiment,
    not a change to that criterion. Unity time-limit endings count as complete games; artificial evaluation safety cutoffs do not.</p>
    <p>The README embeds a short recording of the final student in the actual Unity environment. It is a demonstration,
    separate from the 500-game evaluation.</p>
    ''')
    pages.append(f'''
    <h2>2. Learning algorithm and implementation</h2>
    <p>The teacher learns through shared-policy self-play with clipped Proximal Policy Optimization
    (<a href="https://arxiv.org/abs/1707.06347">Schulman et al., 2017</a>). Each player receives
    [own 24 values, other player’s 24 values]. The actor and critic therefore each use 48 privileged inputs.
    Rewards, episode returns and Generalized Advantage Estimation
    (<a href="https://arxiv.org/abs/1506.02438">Schulman et al., 2015</a>) remain separate for every player and environment.</p>
    {figure('teacher_student_architecture.png', 'Separate teacher actor and critic; the deployed student uses only its own observation and has no critic.')}
    <p>The actor predicts Gaussian means and learns a state-independent log standard deviation. Sampled latent actions pass
    through tanh; action log probabilities include the change-of-variables correction. The critic predicts a state value.</p>
    <div class="equation">PPO loss = −mean[min(rÂ, clip(r, 1−ε, 1+ε)Â)] + 0.5 × mean[(V−R)²]</div>
    <p>Advantages are normalized across the collected batch. True terminals zero the bootstrap value; time limits bootstrap
    the terminal observation. GAE stops at episode boundaries, so it never carries reward estimates into a reset game.</p>
    <h3>Parallel collection</h3>
    <p>Eight workers each hold a frozen copy of the current teacher and collect 512 steps independently. They exchange
    parameters and trajectories once per rollout. The parent combines 8 × 512 × 2 = <strong>8,192 player samples</strong>,
    makes four shuffled minibatch passes, then publishes new parameters. Episodes continue across rollout boundaries.</p>
    <p><code>teacher_model.py</code> defines the networks; <code>teacher_ppo.py</code> implements PPO/GAE;
    <code>teacher_rollouts.py</code> owns worker collection; <code>train_teacher.py</code> logs scores and checkpoints.
    Episode ordering is rollout step then environment index, independent of worker completion order.</p>
    ''')
    pages.append(f'''
    <h2>3. Main experiment: training beyond the solve threshold</h2>
    <p>This experiment restarts from seed 2 with the same initial settings as the successful 512-step run; it is not a
    checkpoint resume. It trains to 1.0 while retaining the first 0.5 crossing and its policy. All other PPO settings remain fixed.</p>
    {table(['Setting', 'Teacher', 'Student'], [
        ('Network', '48 → 128 → 128 → Gaussian(2); critic(1)', '24 → 128 → 128 → 2'),
        ('Hidden/output activations', 'tanh / squashed Gaussian', 'tanh / tanh'),
        ('Optimizer / learning rate', 'Adam / 0.0003', 'Adam / 0.0003'),
        ('Minibatch', '512 player samples', '1,024 examples'),
        ('Collection', '8 environments × 512 steps', 'Offline demonstrations'),
        ('Epochs', '4 per rollout', 'Up to 50; patience 5'),
        ('Discount γ / GAE λ', '0.995 / 0.95', '—'),
        ('Clip ε / value coefficient', '0.2 / 0.5', '—'),
        ('Initial latent log σ', '−0.5', '—'),
        ('Loss', 'Clipped PPO + value MSE', 'Action MSE')])}
    {table(['First training mean', 'Episode', 'Environment steps', 'Observed wall time'], [
        ('.5', f"{half['episode']:,}", f"{half['environment_steps']:,}", duration(half['elapsed_seconds'])),
        ('1.0', f"{one['episode']:,}", f"{one['environment_steps']:,}", duration(one['elapsed_seconds']))])}
    <p>The first 0.5 crossing matches the earlier short-rollout run at 508,160 steps. The extra training to 1.0 uses
    {one['environment_steps']-half['environment_steps']:,} additional environment steps.
    The first qualifying 1.0 window has mean {one['rolling100']:.4f}; the best window over the final rollout reaches
    {teacher['best_rolling100']:.4f}.</p>
    {figure('teacher_extended_training.png', 'The extended learning curve and learned exploration scale. Markers denote first crossings; collection finishes the whole rollout before stopping.')}
    <p class="small">Environment steps sum over all eight games; player transitions are twice that number. The run logs
    {teacher['environment_steps']:,} total steps because the final rollout is completed before scores are checked.
    Wall times record when the completed rollout made each crossing observable.</p>
    ''')
    pages.append(f'''
    <h2>4. Distilling the final teacher</h2>
    {figure('teacher_student_process.png', 'Teacher PPO, fresh demonstration collection, offline student fitting and frozen-policy evaluation are separate stages.')}
    <p>The final teacher is frozen and run deterministically: actions are tanh of the predicted Gaussian means.
    {collection['selected_workers']} independent Unity workers collect <strong>{dataset['samples']:,} player examples</strong>
    from {dataset['episodes']:,} completed games. Both players contribute their current local observation and the matching
    teacher action computed from both players’ observations.</p>
    <p>The split assigns whole matches to training or validation, including both players and all frames.
    There are {student['training_samples']:,} training examples from {student['training_episodes']:,} games and
    {student['validation_samples']:,} validation examples from {student['validation_episodes']:,} games.
    Dataset shards, the manifest and the student config retain teacher fingerprints and checksums.</p>
    <p>A freshly initialized student learns offline with action mean squared error, a direct action-regression form of policy
    distillation (<a href="https://arxiv.org/abs/1511.06295">Rusu et al., 2015</a>). It receives only 24 local inputs and
    requires neither Unity nor rewards during fitting. There is no online student reinforcement learning.</p>
    {figure('student_distillation.png', 'Training and held-out action errors; the selected checkpoint minimizes validation MSE, independently of gameplay scores.')}
    <p>Fitting completes {student['epochs_completed']} epochs in {duration(student['elapsed_seconds'])}, using
    {student['selected_cpu_threads']} CPU threads. The selected epoch is {student['best_epoch']}; validation MSE falls from
    {student['initial_validation_mse']:.5f} to <strong>{student['best_validation_mse']:.5f}</strong>.
    Selected movement/jump MSEs are {float(selected['movement_mse']):.5f}/{float(selected['jump_mse']):.5f},
    with {float(selected['jump_agreement']):.1%} jump-threshold agreement.</p>
    <p>The teacher’s final <code>checkpoint.pt</code>, the student’s final-epoch <code>checkpoint.pt</code>, and the
    validation-selected student <code>best.pt</code> are retained. Evaluation and the GIF use the selected student.</p>
    ''')
    pages.append(f'''
    <h2>5. Frozen-policy results and timing</h2>
    <p>Each policy plays 100 games on each of seeds 10002–10006, with no weight updates. These simulator seeds are
    distinct from training and demonstration seeds. Both players use the same shared policy, each with its appropriate
    observation ordering. All 500 games per policy are included; scores are not selected from favorable windows.</p>
    {table(['Policy', 'Inputs', 'Mean ± SD', 'Passing seeds', 'Unity time limits'], [
        ('Teacher, stochastic', 48, score(stoch), f"{stoch['passed_seeds']}/5", f"{stoch['time_limit_episodes']}/500"),
        ('Teacher, deterministic', 48, score(deterministic), f"{deterministic['passed_seeds']}/5", f"{deterministic['time_limit_episodes']}/500"),
        ('Student, deterministic', 24, score(student_eval), f"{student_eval['passed_seeds']}/5", f"{student_eval['time_limit_episodes']}/500")])}
    {figure('policy_evaluation.png', 'Each dot is a fresh-seed 100-game mean; horizontal marks show the five-seed mean. The dashed line is the official solve threshold.')}
    <p>{comparison} Removing sampling noise is evaluated separately from distillation; the deterministic teacher is
    the relevant action-generating baseline. Evaluation safety cutoffs: teacher stochastic {stoch['truncated_episodes']},
    teacher deterministic {deterministic['truncated_episodes']}, student {student_eval['truncated_episodes']}.</p>
    <p>A 1.0 training window is a stopping event, not a guarantee that the frozen teacher will average at least 1.0
    on every fresh seed. The evaluation above measures that separately.</p>
    {table(['Main pipeline stage', 'Wall time'], [
        ('Teacher: initialization through target 1.0 and shutdown', duration(teacher['elapsed_seconds'])),
        (f"Fresh demonstrations ({collection['selected_workers']} workers)", duration(collection['elapsed_seconds'])),
        ('Offline student fitting, including thread selection', duration(student['elapsed_seconds'])),
        ('Sum of the three stages; excludes evaluation', duration(stage_sum))])}
    <p class="small">Hardware: Intel Core i5-12600K (10 physical cores / 16 logical CPUs), approximately 94 GiB RAM,
    CPU PyTorch 2.14.1, NumPy 1.26.4. No GPU training. Wall times depend on concurrent desktop jobs and are not
    controlled benchmarks. Search runs, the interrupted attempt, evaluation and GIF capture are excluded from the stage sum.</p>
    ''')
    pages.append(f'''
    <h2>6. Earlier experiments and retained comparisons</h2>
    {table(['Experiment', 'Envs × steps/env', 'Epochs / γ', 'Best mean', 'Steps to 0.5'], history_rows)}
    {figure('teacher_training.png', 'Retained teacher trials plotted against aggregate environment steps. Right: raw episode scores and rolling mean of the extended main run.')}
    <p><strong>Initial PPO:</strong> two single-environment runs used 1,000 steps, 10 epochs and γ=0.99, and neither
    solved within one million steps. The adjusted single-environment run used 4,096 steps, four epochs and γ=0.995,
    first solving at 579,040 steps. Those three changes were simultaneous, so their individual effects cannot be separated.</p>
    <p><strong>Parallel implementation:</strong> the original parent selected actions and synchronized workers at every
    simulator step. Moving inference and complete-rollout collection into workers removed that per-step exchange.
    A short equal-batch CPU benchmark improved eight-worker throughput from 1,391 to 2,351 aggregate steps/s;
    the new single-worker result was 690 steps/s. These three-update timings measure execution speed, not learning.</p>
    <p><strong>Rollout comparison:</strong> retaining 4,096 steps per environment enlarged the batch to 65,536 player
    samples and first solved at 1,130,968 steps. Reducing to 512 steps restored 8,192 player samples and solved at 508,160:
    <strong>55.1% fewer steps</strong>. This was also 12.2% fewer than the adjusted single-environment run.
    The stop-at-0.5 short run and the extended main run have the same initial learning curve, not independent evidence.</p>
    {table(['Earlier frozen policy', 'Stochastic teacher', 'Deterministic teacher', 'Student'], [
        ('Adjusted single-env pipeline', '0.537 ± 0.050', score(old_teacher), score(old_student)),
        ('8 env × 4096; stop .5', score(long_stoch), score(long_det), 'Not distilled'),
        ('8 env × 512; stop .5', score(short_stoch), score(short_det), 'Not distilled')])}
    <p class="small">The historical single-environment teacher checkpoint was selected before correcting time-limit
    episode logging; its rewards were re-audited and its frozen checkpoint evaluated directly. An initial target-1 attempt
    received SIGTERM near mean 0.983; its artifacts remain separate. The main result uses a complete seeded rerun.</p>
    ''')
    pages.append(f'''
    <h2>7. Discussion</h2>
    <h3>Why shorter rollouts performed better here</h3>
    <p>The clearest change is <strong>how often learning influences the next collected experience</strong>.
    With eight environments and 4,096 steps each, the policy remains frozen for 32,768 aggregate environment steps.
    At 512 steps, the learner refreshes the policy after 4,096 aggregate steps—eight times sooner. It can therefore
    exploit newly learned returns and adjust its value estimates and exploration before collecting as much additional data.</p>
    <p>Both settings reuse every sample for four epochs with minibatches of 512. A long rollout performs 512 optimizer
    steps per 32,768 environment steps; a short rollout performs 64 per 4,096. Thus optimizer steps per environment
    step and epochs per sample are unchanged. The benefit cannot be described as simply doing more gradient updates.
    The shorter configuration interleaves collection and learning more frequently, and normalizes advantages over a
    smaller, more recent batch. These are plausible explanations, not isolated causal findings.</p>
    <p>Shorter rollouts do not imply incomplete training episodes or forced resets. Games persist across boundaries;
    the critic bootstraps the final observation. They do shorten GAE segments, which can increase dependence on value
    accuracy. Here γλ = 0.94525, corresponding to a decay scale of about 18 steps; this suggests 512 steps need not
    truncate most of the estimator’s weight, although it does not prove that 512 is optimal. The interaction between
    environment count, segment length and batch size is also described by
    <a href="https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/">Huang et al. (2022)</a>.</p>
    <h3>Why the earlier long-rollout adjustment is not contradictory</h3>
    <p>The initial-to-adjusted experiment simultaneously reduced epochs from ten to four and raised γ from .99 to .995.
    It did not demonstrate that long trajectories alone caused learning. The current comparison keeps those improved
    settings and eight environments fixed, while changing rollout length and therefore total batch size.</p>
    <h3>What distillation changes</h3>
    <p>The student approximates deterministic teacher actions using half the inputs and no critic. Regression can smooth
    action variation, but imperfect imitation also changes which states two self-playing students visit. Higher or lower
    returns need not track action MSE monotonically. {comparison} A general claim that students always outperform
    teachers would go beyond these measurements.</p>
    <h3>Does continuing to 1.0 improve the deployed policy?</h3>
    <p>The stop-at-0.5 short-rollout teacher scored {score(short_stoch)} stochastically and
    {score(short_det)} deterministically. After training to 1.0, the corresponding scores are
    {score(stoch)} and {score(deterministic)}. A higher sampled training window need not improve mean-action play:
    these are different action-selection rules, and PPO optimizes returns under sampled actions.
    The final student must be assessed against its actual deterministic teacher and its own gameplay results,
    not inferred from the teacher’s stopping threshold or its imitation error.</p>
    <p>The earlier single-environment pipeline’s student scored {score(old_student)}, compared with
    {score(student_eval)} for the current student. Thus the larger teacher stopping target did not deliver a higher-scoring
    student in these experiments. The new student closely preserves its own teacher’s performance with local inputs;
    the older policies remain available. Different demonstration distributions also mean that lower action MSE across
    datasets cannot by itself rank gameplay quality.</p>
    <h3>Limits and next experiments</h3>
    <p>Most successful configurations have one training seed. The repeated seed-2 short runs do not establish robustness;
    five evaluation seeds measure variation of one frozen policy. Self-play may favor partners with the same behavior,
    and the simulator time limit caps achievable episode returns. Useful next steps are several independent training seeds,
    a fixed-step-budget sweep of 256/512/1,024 steps, KL/clip-fraction diagnostics, and mixed teacher–student partners.</p>
    ''')
    pages.append(f'''
    <h2>8. Reproduction and saved policies</h2>
    <p>The README gives setup, training, evaluation and recording commands. Completed run configs preserve the actual
    settings and are immutable records; fresh prepared reproduction configs create separate runs and datasets.</p>
    {table(['Artifact', 'Location'], [
        ('Main teacher run and all learning logs', main['teacher_run']),
        ('Final teacher used for demonstrations/evaluation', main['teacher_checkpoint']),
        ('First .5 policy snapshot', half['checkpoint']),
        ('First 1.0 policy snapshot', one['checkpoint']),
        ('Demonstration manifest and checksummed shards', main['dataset']),
        ('Student fitting logs and final-epoch checkpoint', main['student_run']),
        ('Selected deployed student', main['student_checkpoint']),
        ('Rendered student gameplay and provenance', 'assets/student_playing.gif; assets/student_playing.json')])}
    <p>Teacher and student checkpoints include reconstruction settings. The student’s selected checkpoint is determined
    only by held-out action MSE. Dataset and evaluation records carry checkpoint hashes so the reported policies can
    be traced to the exact collected examples and evaluation results.</p>
    <h3>References</h3>
    <p>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., &amp; Klimov, O. (2017).
    <em>Proximal Policy Optimization Algorithms.</em> <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a>.</p>
    <p>Schulman, J., Moritz, P., Levine, S., Jordan, M., &amp; Abbeel, P. (2015).
    <em>High-Dimensional Continuous Control Using Generalized Advantage Estimation.</em>
    <a href="https://arxiv.org/abs/1506.02438">arXiv:1506.02438</a>.</p>
    <p>Rusu, A. A., et al. (2015). <em>Policy Distillation.</em>
    <a href="https://arxiv.org/abs/1511.06295">arXiv:1511.06295</a>.</p>
    <p>Huang, S., Dossa, R. F. J., Raffin, A., Kanervisto, A., &amp; Wang, W. (2022).
    <em>The 37 Implementation Details of Proximal Policy Optimization.</em> ICLR Blog Track.
    <a href="https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/">Implementation analysis</a>.</p>
    ''')
    style = '''
    @page { size: A4; margin: 15mm 16mm; }
    * { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; color: #26333e; font-size: 10pt; line-height: 1.48; margin: 0; }
    main { max-width: 920px; margin: auto; }
    section { padding: 30px 36px; background: white; }
    section + section { break-before: page; }
    h1 { font-size: 32pt; line-height: 1.12; color: #244f6b; margin: 12px 0 20px; }
    h2 { color: #244f6b; font-size: 18pt; margin: 0 0 16px; }
    h3 { font-size: 12pt; color: #386e61; margin: 19px 0 6px; }
    p { margin: 9px 0; } .lead { font-size: 13pt; } .eyebrow { text-transform: uppercase; font-size: 8pt; letter-spacing: .1em; color: #526b78; }
    .callout { background: #edf4f4; border-left: 4px solid #487967; padding: 14px 17px; margin: 18px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 8.4pt; margin: 13px 0; }
    th { text-align: left; color: #244f6b; border-bottom: 2px solid #8faebd; padding: 7px 6px; }
    td { padding: 6px; border-bottom: 1px solid #dae3e8; vertical-align: top; overflow-wrap: anywhere; }
    tr { break-inside: avoid; } figure { margin: 16px 0; break-inside: avoid; }
    figure img { display: block; max-width: 100%; max-height: 255px; margin: 0 auto; object-fit: contain; }
    figcaption, .small { font-size: 8.2pt; color: #586976; }
    figcaption { margin-top: 7px; } code { font-size: 8.4pt; color: #3a5666; overflow-wrap: anywhere; }
    a { color: #35658c; } .equation { padding: 10px; background: #f5f7f8; font-size: 9pt; text-align: center; }
    section.discussion { font-size: 9.6pt; line-height: 1.4; }
    section.discussion h3 { margin-top: 14px; }
    @media screen { body { background: #e8edf1; } section { margin: 24px 0; box-shadow: 0 3px 14px #163c4922; } }
    @media print { section { padding: 0; } figure img { max-height: 64mm; } }
    '''
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>PPO Teacher and Local Student for Tennis</title>'
    html += '<style>' + style + '</style><main>'
    html += ''.join(f'<section class="{"discussion" if index == 6 else ""}">{page}</section>'
                    for index, page in enumerate(pages)) + '</main></html>'
    destination = ROOT / 'Report.html'
    destination.write_text(html)
    print(f'Built {destination.name} from completed experiment records.')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()
    source = build_report()
    if args.pdf:
        browser = shutil.which('google-chrome') or shutil.which('chromium')
        if browser is None:
            parser.error('Install Chrome/Chromium to print the report PDF.')
        with tempfile.TemporaryDirectory(prefix='tennis-report-browser-') as profile:
            subprocess.run([browser, '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                            '--allow-file-access-from-files', f'--user-data-dir={profile}',
                            f'--print-to-pdf={ROOT / "Report.pdf"}', source.as_uri()], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('Saved Report.pdf')
