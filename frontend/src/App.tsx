import './App.css'

import {
  useEffect,
  useMemo,
  useState,
  type FormEvent,
  type ReactNode,
} from 'react'

import {
  getAnalyticsOverview,
  getApiHealth,
  getDurationTrend,
  getExecutionHistory,
  getLoaderIntelligence,
  getModelInformation,
  getPredictionHistory,
  getStageAnalytics,
  predictTotalExecution,
  type AnalyticsOverview,
  type DurationTrendPoint,
  type ExecutionHistoryItem,
  type LoaderIntelligenceItem,
  type ModelInformationResponse,
  type PredictionHistoryItem,
  type StageAnalyticsItem,
  type TotalExecutionPredictionResponse,
} from './api'

import { DurationTrendChart } from './components/charts/DurationTrendChart'

type Page =
  | 'overview'
  | 'loader-intelligence'
  | 'loader-detail'
  | 'execution-explorer'
  | 'stage-analytics'
  | 'model-laboratory'
  | 'prediction-studio'
  | 'cold-start'
  | 'prediction-monitoring'
  | 'drift-monitoring'

type NavItem = {
  id: Page
  label: string
  description: string
  icon: string
  badge?: string
}

const NAVIGATION: Array<{
  section: string
  items: NavItem[]
}> = [
  {
    section: 'Monitor',
    items: [
      {
        id: 'overview',
        label: 'Executive Overview',
        description: 'System-wide execution intelligence',
        icon: '⌂',
      },
      {
        id: 'execution-explorer',
        label: 'Execution Explorer',
        description: 'Search and inspect executions',
        icon: '⌕',
      },
      {
        id: 'stage-analytics',
        label: 'Stage Analytics',
        description: 'Stage-level performance',
        icon: '▦',
      },
    ],
  },
  {
    section: 'Intelligence',
    items: [
      {
        id: 'loader-intelligence',
        label: 'Loader Intelligence',
        description: 'Loader behavior and profiles',
        icon: '◈',
      },
      {
        id: 'loader-detail',
        label: 'Loader Detail',
        description: 'Deep loader investigation',
        icon: '◎',
      },
      {
        id: 'prediction-monitoring',
        label: 'Prediction Monitoring',
        description: 'Prediction accuracy and reliability',
        icon: '◔',
      },
      {
        id: 'drift-monitoring',
        label: 'Drift Monitoring',
        description: 'Operational screening for observable change',
        icon: '⌁',
      },
    ],
  },
  {
    section: 'Prediction',
    items: [
      {
        id: 'prediction-studio',
        label: 'Prediction Studio',
        description: 'Generate execution-time estimates',
        icon: '✦',
      },
      {
        id: 'cold-start',
        label: 'Cold Start Intelligence',
        description: 'Unseen-loader intelligence',
        icon: '◇',
        badge: 'M3',
      },
    ],
  },
  {
    section: 'Modeling',
    items: [
      {
        id: 'model-laboratory',
        label: 'Model Laboratory',
        description: 'Model comparison and rationale',
        icon: '△',
        badge: 'M2',
      },
    ],
  },
]

const PAGE_META: Record<
  Page,
  {
    eyebrow: string
    title: string
    description: string
  }
> = {
  overview: {
    eyebrow: 'Command Center',
    title: 'Execution Intelligence',
    description:
      'A unified view of Data Loader execution behavior, prediction performance, and operational signals.',
  },
  'loader-intelligence': {
    eyebrow: 'Module 1',
    title: 'Loader Intelligence',
    description:
      'Understand loader behavior, historical execution patterns, workload characteristics, and stability.',
  },
  'loader-detail': {
    eyebrow: 'Loader Intelligence',
    title: 'Loader Detail',
    description:
      'Investigate an individual loader across history, configuration, stages, prediction accuracy, and evidence.',
  },
  'execution-explorer': {
    eyebrow: 'Module 1',
    title: 'Execution Explorer',
    description:
      'Search and investigate historical Data Loader executions at execution level.',
  },
  'stage-analytics': {
    eyebrow: 'Module 1',
    title: 'Stage Analytics',
    description:
      'Analyze stage timing distributions, observations, outliers, and operational contribution.',
  },
  'model-laboratory': {
    eyebrow: 'Module 2',
    title: 'Model Laboratory',
    description:
      'Review the deployed model, its configuration, production rationale, and known limitations.',
  },
  'prediction-studio': {
    eyebrow: 'Module 2',
    title: 'Prediction Studio',
    description:
      'Generate a pre-execution estimate using only information available before execution begins.',
  },
  'cold-start': {
    eyebrow: 'Module 3',
    title: 'Cold Start Intelligence',
    description:
      'Understand how the platform estimates execution time when a loader has no prior eligible history.',
  },
  'prediction-monitoring': {
    eyebrow: 'Operational Monitoring',
    title: 'Prediction Monitoring',
    description:
      'Track prediction accuracy, known-versus-unseen behavior, reliability, and observed error.',
  },
  'drift-monitoring': {
    eyebrow: 'Operational Monitoring',
    title: 'Drift Monitoring',
    description:
      'Monitor observable changes in workload, loaders, stages, configuration coverage, and execution behavior.',
  },
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—'
  }

  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
  }).format(value)
}

function formatDecimal(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—'
  }

  return value.toFixed(digits)
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return '—'
  }

  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 1 : 0)} sec`
  }

  if (seconds < 3600) {
    return `${(seconds / 60).toFixed(1)} min`
  }

  if (seconds < 86400) {
    return `${(seconds / 3600).toFixed(1)} hr`
  }

  return `${(seconds / 86400).toFixed(1)} days`
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return '—'
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function statusTone(status: string | null | undefined): string {
  const normalized = status?.toLowerCase() ?? ''

  if (
    normalized.includes('succeed') ||
    normalized.includes('success') ||
    normalized.includes('complete')
  ) {
    return 'status-success'
  }

  if (
    normalized.includes('stop') ||
    normalized.includes('fail') ||
    normalized.includes('cancel') ||
    normalized.includes('error')
  ) {
    return 'status-danger'
  }

  if (
    normalized.includes('warn') ||
    normalized.includes('running') ||
    normalized.includes('progress')
  ) {
    return 'status-warning'
  }

  return 'status-neutral'
}

function sourceTone(source: string | null | undefined): string {
  const normalized = source?.toLowerCase() ?? ''

  if (
    normalized.includes('global') ||
    normalized.includes('cold') ||
    normalized.includes('unseen')
  ) {
    return 'status-warning'
  }

  if (
    normalized.includes('loader') ||
    normalized.includes('known')
  ) {
    return 'status-success'
  }

  return 'status-neutral'
}

function StatCard({
  label,
  value,
  detail,
  tone = 'default',
  icon,
}: {
  label: string
  value: string
  detail?: string
  tone?: 'default' | 'positive' | 'warning' | 'danger'
  icon?: string
}) {
  return (
    <article className={`stat-card stat-card-${tone}`}>
      <div className="stat-card-top">
        <span className="stat-label">{label}</span>
        {icon ? (
          <span className="stat-icon" aria-hidden="true">
            {icon}
          </span>
        ) : (
          <span className="stat-dot" />
        )}
      </div>

      <div className="stat-value">{value}</div>

      {detail && <div className="stat-detail">{detail}</div>}
    </article>
  )
}

function SectionCard({
  title,
  subtitle,
  children,
  action,
}: {
  title: string
  subtitle?: string
  children: ReactNode
  action?: ReactNode
}) {
  return (
    <section className="section-card">
      <div className="section-card-header">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>

        {action}
      </div>

      {children}
    </section>
  )
}

function EmptyState({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">◌</div>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  )
}

function InlineSignal({
  label,
  value,
  tone = 'neutral',
}: {
  label: string
  value: string
  tone?: 'success' | 'warning' | 'danger' | 'neutral'
}) {
  return (
    <div className={`inline-signal inline-signal-${tone}`}>
      <span />
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
      </div>
    </div>
  )
}

function loadersWithHistoryFromPredictions(
  predictions: PredictionHistoryItem[],
): string {
  const names = new Set(
    predictions
      .filter((item) => {
        const source = item.used_prediction_source?.toLowerCase() ?? ''
        return source.includes('loader') || source.includes('known')
      })
      .map((item) => item.loader_name)
      .filter((name): name is string => Boolean(name)),
  )

  return formatNumber(names.size)
}

function OverviewPage({
  overview,
  executions,
  predictions,
  durationTrend,
  model,
  onNavigate,
}: {
  overview: AnalyticsOverview | null
  executions: ExecutionHistoryItem[]
  predictions: PredictionHistoryItem[]
  durationTrend: DurationTrendPoint[]
  model: ModelInformationResponse | null
  onNavigate: (page: Page) => void
}) {
  const predictionMetrics = useMemo(() => {
    const errors = predictions
      .map((item) => item.absolute_error_seconds)
      .filter(
        (value): value is number =>
          value !== null && Number.isFinite(value),
      )

    const relativeErrors = predictions
      .map((item) => item.relative_error)
      .filter(
        (value): value is number =>
          value !== null && Number.isFinite(value),
      )

    const withinTwentyPercent = relativeErrors.filter(
      (value) => Math.abs(value) <= 0.2,
    )

    const known = predictions.filter((item) => {
      const source = item.used_prediction_source?.toLowerCase() ?? ''
      return source.includes('loader') || source.includes('known')
    })

    const unseen = predictions.filter((item) => {
      const source = item.used_prediction_source?.toLowerCase() ?? ''
      return (
        source.includes('global') ||
        source.includes('cold') ||
        source.includes('unseen')
      )
    })

    const mae = errors.length
      ? errors.reduce((sum, value) => sum + value, 0) / errors.length
      : null

    return {
      count: predictions.length,
      mae,
      knownCount: known.length,
      unseenCount: unseen.length,
      observedRelativeErrors: relativeErrors.length,
      withinTwentyPercent:
        relativeErrors.length > 0
          ? (withinTwentyPercent.length / relativeErrors.length) * 100
          : null,
    }
  }, [predictions])

  const latestExecutions = executions.slice(0, 6)
  const latestPrediction = predictions[0] ?? null

  const reliabilityLabel =
    predictionMetrics.withinTwentyPercent === null
      ? 'Not measurable'
      : predictionMetrics.withinTwentyPercent >= 80
        ? 'High'
        : predictionMetrics.withinTwentyPercent >= 60
          ? 'Moderate'
          : 'Developing'

  return (
    <div className="page-stack">
      <div className="metric-grid metric-grid-eight">
        <StatCard
          label="Total executions"
          value={formatNumber(overview?.total_executions)}
          detail="Historical execution records"
        />

        <StatCard
          label="Successful"
          value={formatNumber(overview?.successful_executions)}
          detail="Observed successful paths"
          tone="positive"
        />

        <StatCard
          label="Active loaders"
          value={formatNumber(overview?.unique_loaders)}
          detail="Distinct historical loaders"
        />

        <StatCard
          label="Known-loader evidence"
          value={
            loadersWithHistoryFromPredictions(predictions)
          }
          detail="Prediction records routed through loader history"
          tone="positive"
        />

        <StatCard
          label="Predictions"
          value={formatNumber(predictionMetrics.count)}
          detail="Persisted prediction records"
        />

        <StatCard
          label="Avg prediction error"
          value={formatDuration(predictionMetrics.mae)}
          detail="Observed feedback only"
        />

        <StatCard
          label="Prediction reliability"
          value={reliabilityLabel}
          detail={
            predictionMetrics.withinTwentyPercent === null
              ? 'Needs actual-duration feedback'
              : `${formatDecimal(
                  predictionMetrics.withinTwentyPercent,
                  1,
                )}% within ±20%`
          }
          tone={
            reliabilityLabel === 'High'
              ? 'positive'
              : reliabilityLabel === 'Moderate'
                ? 'warning'
                : 'default'
          }
        />

        <StatCard
          label="High-risk executions"
          value="—"
          detail="Risk model is not part of Modules 1–3"
          tone="warning"
        />
      </div>

      <div className="dashboard-grid dashboard-grid-main">
        <SectionCard
          title="Operating picture"
          subtitle="Prediction population and observed feedback"
          action={
            <button
              className="text-button"
              onClick={() => onNavigate('prediction-monitoring')}
            >
              Monitor predictions →
            </button>
          }
        >
          <div className="operating-picture">
            <div className="operating-primary">
              <span>Known-loader predictions</span>
              <strong>{formatNumber(predictionMetrics.knownCount)}</strong>
              <small>Historical loader evidence</small>
            </div>

            <div className="operating-stat">
              <span>Cold-start predictions</span>
              <strong>{formatNumber(predictionMetrics.unseenCount)}</strong>
              <small>Global fallback route</small>
            </div>

            <div className="operating-stat">
              <span>Observed actuals</span>
              <strong>{formatNumber(predictionMetrics.observedRelativeErrors)}</strong>
              <small>With relative-error feedback</small>
            </div>

            <div className="operating-stat">
              <span>Unobserved predictions</span>
              <strong>
                {formatNumber(
                  Math.max(
                    predictionMetrics.count -
                      predictionMetrics.observedRelativeErrors,
                    0,
                  ),
                )}
              </strong>
              <small>Actual duration not yet available</small>
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Execution profile"
          subtitle="Historical duration context from completed executions"
          action={
            <button
              className="text-button"
              onClick={() => onNavigate('execution-explorer')}
            >
              Explore executions →
            </button>
          }
        >
          <div className="operating-picture">
            <div className="operating-primary">
              <span>Median duration</span>
              <strong>{formatDuration(overview?.median_duration_seconds)}</strong>
              <small>P90 {formatDuration(overview?.p90_duration_seconds)}</small>
            </div>

            <div className="operating-stat">
              <span>Average</span>
              <strong>{formatDuration(overview?.average_duration_seconds)}</strong>
            </div>

            <div className="operating-stat">
              <span>Maximum</span>
              <strong>{formatDuration(overview?.maximum_duration_seconds)}</strong>
            </div>

            <div className="operating-stat">
              <span>Total records</span>
              <strong>{formatNumber(overview?.total_records)}</strong>
            </div>
          </div>
        </SectionCard>
      </div>

      <div className="dashboard-grid dashboard-grid-main">
        <SectionCard
          title="Module status"
          subtitle="Verified architecture currently available"
        >
          <div className="architecture-list">
            <div className="architecture-item">
              <span className="architecture-number">01</span>
              <div>
                <strong>Historical execution intelligence</strong>
                <small>
                  {formatNumber(overview?.total_executions)} execution
                  records and {formatNumber(overview?.unique_loaders)} loaders
                  are available through analytics.
                </small>
              </div>
              <span className="architecture-live">LIVE</span>
            </div>

            <div className="architecture-item">
              <span className="architecture-number">02</span>
              <div>
                <strong>Execution-time prediction</strong>
                <small>
                  {model?.estimator_type ?? 'ExtraTreesRegressor'} ·{' '}
                  {model?.model_version ?? 'total-et-v2'} · log1p target
                </small>
              </div>
              <span className="architecture-live">LIVE</span>
            </div>

            <div className="architecture-item">
              <span className="architecture-number">03</span>
              <div>
                <strong>Cold-start intelligence</strong>
                <small>
                  Global historical median is the primary unseen-loader
                  fallback.
                </small>
              </div>
              <span className="architecture-live">LIVE</span>
            </div>

            <div className="architecture-item">
              <span className="architecture-number">04</span>
              <div>
                <strong>Supporting similarity evidence</strong>
                <small>
                  Similar loaders are retained as evidence and are not the
                  primary prediction route.
                </small>
              </div>
              <span className="architecture-live">LIVE</span>
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Latest prediction"
          subtitle="Most recent persisted prediction feedback"
          action={
            <button
              className="text-button"
              onClick={() => onNavigate('prediction-studio')}
            >
              Open Prediction Studio →
            </button>
          }
        >
          {latestPrediction ? (
            <div className="detail-grid">
              <div>
                <span>Execution</span>
                <strong className="mono">
                  {latestPrediction.execution_id}
                </strong>
              </div>

              <div>
                <span>Loader</span>
                <strong>{latestPrediction.loader_name ?? '—'}</strong>
              </div>

              <div>
                <span>Predicted</span>
                <strong>
                  {formatDuration(
                    latestPrediction.predicted_total_seconds,
                  )}
                </strong>
              </div>

              <div>
                <span>Actual</span>
                <strong>
                  {formatDuration(
                    latestPrediction.actual_total_seconds,
                  )}
                </strong>
              </div>

              <div>
                <span>Source</span>
                <strong>
                  {latestPrediction.used_prediction_source ?? '—'}
                </strong>
              </div>

              <div>
                <span>Model</span>
                <strong>
                  {latestPrediction.model_version ?? '—'}
                </strong>
              </div>
            </div>
          ) : (
            <EmptyState
              title="No persisted prediction feedback"
              description="Generate a prediction and evaluate it against a completed execution before accuracy metrics become available."
            />
          )}
        </SectionCard>
      </div>

      <div className="dashboard-grid dashboard-grid-main">
        <SectionCard
          title="Execution duration trend"
          subtitle="Historical daily average and median execution duration"
        >
          <DurationTrendChart points={durationTrend} />
        </SectionCard>

        <SectionCard
          title="Prediction vs actual"
          subtitle="Latest persisted record with observed feedback"
        >
          {latestPrediction?.predicted_total_seconds !== null &&
          latestPrediction?.actual_total_seconds !== null &&
          latestPrediction?.predicted_total_seconds !== undefined &&
          latestPrediction?.actual_total_seconds !== undefined ? (
            <div className="prediction-actual-visual">
              <div className="prediction-actual-summary">
                <div>
                  <span>Predicted</span>
                  <strong>
                    {formatDuration(latestPrediction.predicted_total_seconds)}
                  </strong>
                </div>
                <div>
                  <span>Actual</span>
                  <strong>
                    {formatDuration(latestPrediction.actual_total_seconds)}
                  </strong>
                </div>
              </div>

              <div className="comparison-bars" aria-label="Predicted versus actual duration">
                <div className="comparison-row">
                  <span>Prediction</span>
                  <div className="comparison-track">
                    <span
                      className="comparison-fill comparison-fill-prediction"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            4,
                            (latestPrediction.predicted_total_seconds /
                              Math.max(
                                latestPrediction.predicted_total_seconds,
                                latestPrediction.actual_total_seconds,
                                1,
                              )) *
                              100,
                          ),
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="comparison-row">
                  <span>Actual</span>
                  <div className="comparison-track">
                    <span
                      className="comparison-fill comparison-fill-actual"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            4,
                            (latestPrediction.actual_total_seconds /
                              Math.max(
                                latestPrediction.predicted_total_seconds,
                                latestPrediction.actual_total_seconds,
                                1,
                              )) *
                              100,
                          ),
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              </div>

              <small className="comparison-note">
                Latest record: {latestPrediction.execution_id}
              </small>
            </div>
          ) : (
            <EmptyState
              title="No observed prediction pair"
              description="Predicted and actual durations must both be available before this comparison can be shown."
            />
          )}
        </SectionCard>
      </div>

      <SectionCard
        title="Recent execution activity"
        subtitle="Latest records available through the analytics API"
        action={
          <button
            className="text-button"
            onClick={() => onNavigate('execution-explorer')}
          >
            View all →
          </button>
        }
      >
        {latestExecutions.length ? (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Execution</th>
                  <th>Loader</th>
                  <th>Status</th>
                  <th>Records</th>
                  <th>Duration</th>
                  <th>Started</th>
                </tr>
              </thead>

              <tbody>
                {latestExecutions.map((execution) => (
                  <tr key={execution.execution_id}>
                    <td>
                      <span className="mono">
                        {execution.execution_id}
                      </span>
                    </td>

                    <td>
                      <strong>
                        {execution.loader_name ??
                          'Unidentified loader'}
                      </strong>
                    </td>

                    <td>
                      <span
                        className={`status-pill ${statusTone(
                          execution.status,
                        )}`}
                      >
                        {execution.status ?? 'Unknown'}
                      </span>
                    </td>

                    <td>{formatNumber(execution.total_records)}</td>

                    <td>
                      {formatDuration(execution.duration_seconds)}
                    </td>

                    <td>{formatDateTime(execution.start_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No execution activity"
            description="The analytics API has not returned execution records yet."
          />
        )}
      </SectionCard>
    </div>
  )
}

function LoaderIntelligencePage({
  loaders,
  onSelect,
}: {
  loaders: LoaderIntelligenceItem[]
  onSelect: (loader: string) => void
}) {
  const [search, setSearch] = useState('')

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()

    const results = !query
      ? loaders
      : loaders.filter((loader) =>
          loader.loader_name.toLowerCase().includes(query),
        )

    return [...results].sort(
      (a, b) => b.execution_count - a.execution_count,
    )
  }, [loaders, search])

  const totalExecutions = loaders.reduce(
    (sum, loader) => sum + loader.execution_count,
    0,
  )

  const totalRecords = loaders.reduce(
    (sum, loader) => sum + loader.total_records,
    0,
  )

  const stableLoaders = loaders.filter(
    (loader) => loader.execution_count >= 3,
  ).length

  return (
    <div className="page-stack">
      <div className="metric-grid metric-grid-four">
        <StatCard
          label="Loader population"
          value={formatNumber(loaders.length)}
          detail="Distinct historical loaders"
        />

        <StatCard
          label="Historical executions"
          value={formatNumber(totalExecutions)}
          detail="Across all profiled loaders"
        />

        <StatCard
          label="Historical records"
          value={formatNumber(totalRecords)}
          detail="Observed workload volume"
        />

        <StatCard
          label="Established loaders"
          value={formatNumber(stableLoaders)}
          detail="Loaders with at least 3 observations"
        />
      </div>

      <div className="toolbar-card">
        <div>
          <span className="section-eyebrow">
            Loader intelligence
          </span>

          <h2>
            {formatNumber(filtered.length)} loaders in view
          </h2>

          <p>
            Search historical loader profiles and inspect their execution
            behavior.
          </p>
        </div>

        <label className="search-field">
          <span>⌕</span>

          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search loader..."
            aria-label="Search loader"
          />
        </label>
      </div>

      <SectionCard
        title="Loader profiles"
        subtitle="Historical execution statistics and workload footprint"
      >
        {filtered.length ? (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Loader</th>
                  <th>Executions</th>
                  <th>Successful</th>
                  <th>Median</th>
                  <th>P90</th>
                  <th>Maximum</th>
                  <th>Records</th>
                  <th>Latest execution</th>
                  <th />
                </tr>
              </thead>

              <tbody>
                {filtered.map((loader) => (
                  <tr key={loader.loader_name}>
                    <td>
                      <div>
                        <strong>{loader.loader_name}</strong>
                        <small className="table-secondary">
                          Historical profile
                        </small>
                      </div>
                    </td>

                    <td>
                      <strong>
                        {formatNumber(loader.execution_count)}
                      </strong>
                    </td>

                    <td>
                      {formatNumber(
                        loader.successful_execution_count,
                      )}
                    </td>

                    <td>
                      {formatDuration(
                        loader.median_duration_seconds,
                      )}
                    </td>

                    <td>
                      {formatDuration(
                        loader.p90_duration_seconds,
                      )}
                    </td>

                    <td>
                      {formatDuration(
                        loader.maximum_duration_seconds,
                      )}
                    </td>

                    <td>{formatNumber(loader.total_records)}</td>

                    <td>
                      {formatDateTime(loader.latest_execution)}
                    </td>

                    <td>
                      <button
                        className="secondary-button"
                        onClick={() =>
                          onSelect(loader.loader_name)
                        }
                      >
                        Inspect →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title={
              search.trim()
                ? 'No matching loaders'
                : 'No loader history'
            }
            description={
              search.trim()
                ? 'Try another loader name or clear the search.'
                : 'The analytics API has not returned historical loader profiles yet.'
            }
          />
        )}
      </SectionCard>
    </div>
  )
}

function LoaderDetailPage({
  loader,
  stages,
  predictions,
  model,
  onBack,
  onPredict,
  onStageAnalytics,
}: {
  loader: LoaderIntelligenceItem | null
  stages: StageAnalyticsItem[]
  predictions: PredictionHistoryItem[]
  model: ModelInformationResponse | null
  onBack: () => void
  onPredict: () => void
  onStageAnalytics: () => void
}) {
  if (!loader) {
    return (
      <div className="page-stack">
        <EmptyState
          title="Select a loader first"
          description="Open Loader Intelligence and select a loader to inspect its detailed profile."
        />
      </div>
    )
  }

  const loaderPredictions = predictions.filter(
    (item) => item.loader_name === loader.loader_name,
  )

  const observedPredictions = loaderPredictions.filter(
    (item) =>
      item.absolute_error_seconds !== null &&
      Number.isFinite(item.absolute_error_seconds),
  )

  const relativeErrors = loaderPredictions
    .map((item) => item.relative_error)
    .filter(
      (value): value is number =>
        value !== null && Number.isFinite(value),
    )

  const withinTwenty =
    relativeErrors.length > 0
      ? (relativeErrors.filter(
          (value) => Math.abs(value) <= 0.2,
        ).length /
          relativeErrors.length) *
        100
      : null

  const loaderMae = observedPredictions.length
    ? observedPredictions.reduce(
        (sum, item) =>
          sum + (item.absolute_error_seconds ?? 0),
        0,
      ) / observedPredictions.length
    : null

  const successRate =
    loader.execution_count > 0
      ? (loader.successful_execution_count /
          loader.execution_count) *
        100
      : null

  return (
    <div className="page-stack">
      <div className="loader-detail-hero">
        <div>
          <span className="section-eyebrow">Known loader</span>

          <h2>{loader.loader_name}</h2>

          <p>
            Historical behavior, workload footprint, prediction readiness,
            and observed feedback for this loader.
          </p>
        </div>

        <div className="loader-detail-actions">
          <button
            className="secondary-button"
            onClick={onBack}
          >
            ← Back to loaders
          </button>

          <button
            className="primary-button"
            onClick={onPredict}
          >
            Predict execution time →
          </button>
        </div>
      </div>

      <div className="metric-grid metric-grid-six">
        <StatCard
          label="Executions"
          value={formatNumber(loader.execution_count)}
          detail="Historical observations"
        />

        <StatCard
          label="Successful"
          value={formatNumber(loader.successful_execution_count)}
          detail={
            successRate === null
              ? 'Observed successful paths'
              : `${formatDecimal(successRate, 1)}% of executions`
          }
          tone="positive"
        />

        <StatCard
          label="Median"
          value={formatDuration(
            loader.median_duration_seconds,
          )}
          detail="Typical duration"
        />

        <StatCard
          label="P90"
          value={formatDuration(loader.p90_duration_seconds)}
          detail="Upper-tail duration"
        />

        <StatCard
          label="Maximum"
          value={formatDuration(
            loader.maximum_duration_seconds,
          )}
          detail="Observed maximum"
          tone="warning"
        />

        <StatCard
          label="Records"
          value={formatNumber(loader.total_records)}
          detail="Historical workload volume"
        />
      </div>

      <div className="dashboard-grid dashboard-grid-main">
        <SectionCard
          title="Loader profile"
          subtitle="Current evidence available for this loader"
        >
          <div className="detail-grid">
            <div>
              <span>Loader classification</span>
              <strong>Known loader</strong>
            </div>

            <div>
              <span>Historical executions</span>
              <strong>
                {formatNumber(loader.execution_count)}
              </strong>
            </div>

            <div>
              <span>Successful executions</span>
              <strong>
                {formatNumber(
                  loader.successful_execution_count,
                )}
              </strong>
            </div>

            <div>
              <span>Latest execution</span>
              <strong>
                {formatDateTime(loader.latest_execution)}
              </strong>
            </div>

            <div>
              <span>Prediction model</span>
              <strong>
                {model?.model_version ?? 'total-et-v2'}
              </strong>
            </div>

            <div>
              <span>Prediction route</span>
              <strong>Known-loader model</strong>
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Prediction readiness"
          subtitle="Historical evidence supports the known-loader route"
        >
          <div className="readiness-panel">
            <div className="readiness-status">
              <span className="readiness-indicator" />

              <div>
                <strong>Historical evidence available</strong>
                <small>
                  The loader has a historical profile and can be routed
                  through the known-loader prediction path.
                </small>
              </div>
            </div>

            <div className="readiness-list">
              <div>
                <span>Model</span>
                <strong>
                  {model?.model_version ?? 'total-et-v2'}
                </strong>
              </div>

              <div>
                <span>Estimator</span>
                <strong>
                  {model?.estimator_type ?? 'ExtraTreesRegressor'}
                </strong>
              </div>

              <div>
                <span>Target</span>
                <strong>Total execution duration</strong>
              </div>

              <div>
                <span>Prediction boundary</span>
                <strong>Before execution begins</strong>
              </div>
            </div>

            <button
              className="primary-button readiness-button"
              onClick={onPredict}
            >
              Configure prediction →
            </button>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Historical behavior"
        subtitle="Loader-level execution distribution"
      >
        <div className="profile-stat-grid">
          <div className="profile-stat">
            <span>Median duration</span>
            <strong>
              {formatDuration(loader.median_duration_seconds)}
            </strong>
            <small>Central historical tendency</small>
          </div>

          <div className="profile-stat">
            <span>P90 duration</span>
            <strong>
              {formatDuration(loader.p90_duration_seconds)}
            </strong>
            <small>Higher-duration historical range</small>
          </div>

          <div className="profile-stat">
            <span>Maximum duration</span>
            <strong>
              {formatDuration(loader.maximum_duration_seconds)}
            </strong>
            <small>Longest observed execution</small>
          </div>

          <div className="profile-stat">
            <span>Average duration</span>
            <strong>
              {formatDuration(loader.average_duration_seconds)}
            </strong>
            <small>Historical arithmetic average</small>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title="Stage behavior"
        subtitle="Population-level stage evidence available from the analytics API"
        action={
          <button
            className="text-button"
            onClick={onStageAnalytics}
          >
            Open Stage Analytics →
          </button>
        }
      >
        {stages.length ? (
          <div className="stage-intelligence-grid">
            {stages.map((stage) => (
              <div
                className="stage-intelligence-card"
                key={stage.stage_name}
              >
                <div className="stage-intelligence-top">
                  <span>{stage.stage_name}</span>
                  <span>{stage.unit}</span>
                </div>

                <strong>
                  {formatDecimal(stage.median_duration, 2)}
                  <small> median</small>
                </strong>

                <div className="stage-intelligence-meta">
                  <span>
                    P90 {formatDecimal(stage.p90_duration, 2)}
                  </span>

                  <span>
                    {formatNumber(stage.observations)} obs.
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No stage behavior available"
            description="Stage analytics are not currently available."
          />
        )}
      </SectionCard>

      <SectionCard
        title="Prediction accuracy"
        subtitle="Only observed actual-duration feedback is used here"
      >
        {loaderPredictions.length ? (
          <div className="prediction-accuracy-panel">
            <div className="accuracy-primary">
              <span>Predictions</span>
              <strong>
                {formatNumber(loaderPredictions.length)}
              </strong>
              <small>Persisted predictions for this loader</small>
            </div>

            <div>
              <span>Observed MAE</span>
              <strong>{formatDuration(loaderMae)}</strong>
              <small>Actual-duration feedback only</small>
            </div>

            <div>
              <span>Within ±20%</span>
              <strong>
                {withinTwenty === null
                  ? '—'
                  : `${formatDecimal(withinTwenty, 1)}%`}
              </strong>
              <small>Observed relative errors</small>
            </div>

            <div>
              <span>Latest route</span>
              <strong>
                {loaderPredictions[0]
                  ?.used_prediction_source ?? '—'}
              </strong>
              <small>Prediction routing evidence</small>
            </div>
          </div>
        ) : (
          <EmptyState
            title="No prediction feedback for this loader"
            description="Generate a prediction and compare it with the resulting execution to establish loader-specific accuracy evidence."
          />
        )}
      </SectionCard>

      <SectionCard
        title="Operational interpretation"
        subtitle="Evidence is separated from unsupported risk claims"
      >
        <div className="evidence-grid">
          <div className="evidence-item">
            <span className="evidence-icon">01</span>
            <div>
              <strong>Historical evidence</strong>
              <p>
                {formatNumber(loader.execution_count)} executions are
                available in the loader profile.
              </p>
            </div>
          </div>

          <div className="evidence-item">
            <span className="evidence-icon">02</span>
            <div>
              <strong>Workload footprint</strong>
              <p>
                {formatNumber(loader.total_records)} historical records
                are associated with this loader.
              </p>
            </div>
          </div>

          <div className="evidence-item">
            <span className="evidence-icon">03</span>
            <div>
              <strong>Prediction route</strong>
              <p>
                Historical loader evidence allows the known-loader model
                route rather than cold-start fallback.
              </p>
            </div>
          </div>

          <div className="evidence-item">
            <span className="evidence-icon">04</span>
            <div>
              <strong>Risk boundary</strong>
              <p>
                Timing statistics alone are not presented as a calibrated
                execution-risk score.
              </p>
            </div>
          </div>
        </div>
      </SectionCard>
    </div>
  )
}

function ExecutionExplorerPage({
  executions,
}: {
  executions: ExecutionHistoryItem[]
}) {
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('ALL')
  const [durationBand, setDurationBand] = useState('ALL')

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()

    return executions.filter((execution) => {
      const matchesSearch =
        !query ||
        execution.execution_id.toLowerCase().includes(query) ||
        (execution.loader_name ?? '')
          .toLowerCase()
          .includes(query) ||
        (execution.sprint ?? '')
          .toLowerCase()
          .includes(query)

      const normalizedStatus =
        execution.status?.toUpperCase() ?? ''

      const matchesStatus =
        status === 'ALL' ||
        normalizedStatus.includes(status)

      const duration =
        execution.duration_seconds ?? null

      const matchesDuration =
        durationBand === 'ALL'
          ? true
          : durationBand === 'SHORT'
            ? duration !== null && duration < 60
            : durationBand === 'MEDIUM'
              ? duration !== null &&
                duration >= 60 &&
                duration < 3600
              : duration !== null && duration >= 3600

      return (
        matchesSearch &&
        matchesStatus &&
        matchesDuration
      )
    })
  }, [executions, search, status, durationBand])

  return (
    <div className="page-stack">
      <div className="toolbar-card">
        <div>
          <span className="section-eyebrow">
            Execution search
          </span>

          <h2>
            {formatNumber(filtered.length)} matching executions
          </h2>

          <p>
            Search by execution ID, loader, or sprint and filter the
            historical execution population.
          </p>
        </div>

        <div className="toolbar-controls">
          <label className="search-field">
            <span>⌕</span>

            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search executions..."
              aria-label="Search executions"
            />
          </label>

          <select
            className="select-control"
            value={status}
            onChange={(event) => setStatus(event.target.value)}
          >
            <option value="ALL">All statuses</option>
            <option value="SUCCEED">Succeed</option>
            <option value="STOP">Stopped</option>
            <option value="FAIL">Failed</option>
            <option value="ERROR">Error</option>
          </select>

          <select
            className="select-control"
            value={durationBand}
            onChange={(event) =>
              setDurationBand(event.target.value)
            }
          >
            <option value="ALL">All durations</option>
            <option value="SHORT">&lt; 1 min</option>
            <option value="MEDIUM">1 min – 1 hr</option>
            <option value="LONG">&gt; 1 hr</option>
          </select>
        </div>
      </div>

      <SectionCard
        title="Execution Explorer"
        subtitle="Execution → workload → stage timing"
      >
        {filtered.length ? (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Execution ID</th>
                  <th>Loader</th>
                  <th>Sprint</th>
                  <th>Status</th>
                  <th>Datasets</th>
                  <th>Records</th>
                  <th>Duration</th>
                  <th>Stages</th>
                </tr>
              </thead>

              <tbody>
                {filtered.map((execution) => (
                  <tr key={execution.execution_id}>
                    <td>
                      <span className="mono">
                        {execution.execution_id}
                      </span>
                    </td>

                    <td>{execution.loader_name ?? '—'}</td>

                    <td>{execution.sprint ?? '—'}</td>

                    <td>
                      <span
                        className={`status-pill ${statusTone(
                          execution.status,
                        )}`}
                      >
                        {execution.status ?? '—'}
                      </span>
                    </td>

                    <td>
                      {formatNumber(execution.dataset_count)}
                    </td>

                    <td>
                      {formatNumber(execution.total_records)}
                    </td>

                    <td>
                      {formatDuration(
                        execution.duration_seconds,
                      )}
                    </td>

                    <td>
                      <span className="stage-summary">
                        P{' '}
                        {formatDuration(
                          execution.prevalidation_duration_seconds,
                        )}
                        <br />
                        T{' '}
                        {formatDuration(
                          execution.transformation_duration_seconds,
                        )}
                        <br />
                        L{' '}
                        {formatDuration(
                          execution.data_loading_duration_seconds,
                        )}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No executions found"
            description="Adjust the search or filters."
          />
        )}
      </SectionCard>
    </div>
  )
}

function StageAnalyticsPage({
  stages,
}: {
  stages: StageAnalyticsItem[]
}) {
  const mostObserved = [...stages].sort(
    (a, b) => b.observations - a.observations,
  )[0]

  const highestP90 = [...stages]
    .filter((stage) => stage.p90_duration !== null)
    .sort(
      (a, b) =>
        (b.p90_duration ?? 0) - (a.p90_duration ?? 0),
    )[0]

  const totalObservations = stages.reduce(
    (sum, stage) => sum + stage.observations,
    0,
  )

  return (
    <div className="page-stack">
      <div className="metric-grid metric-grid-four">
        <StatCard
          label="Stages observed"
          value={formatNumber(stages.length)}
          detail="Distinct stage categories"
        />

        <StatCard
          label="Total observations"
          value={formatNumber(totalObservations)}
          detail="Across all returned stages"
        />

        <StatCard
          label="Most observed"
          value={mostObserved?.stage_name ?? '—'}
          detail={
            mostObserved
              ? `${formatNumber(
                  mostObserved.observations,
                )} observations`
              : 'No stage data'
          }
        />

        <StatCard
          label="Highest P90"
          value={formatDuration(
            highestP90?.p90_duration,
          )}
          detail={highestP90?.stage_name ?? 'No stage data'}
          tone="warning"
        />
      </div>

      <SectionCard
        title="Stage performance"
        subtitle="Observed historical stage timing"
      >
        {stages.length ? (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Unit</th>
                  <th>Observations</th>
                  <th>Median</th>
                  <th>Average</th>
                  <th>P90</th>
                  <th>Maximum</th>
                </tr>
              </thead>

              <tbody>
                {stages.map((stage) => (
                  <tr key={stage.stage_name}>
                    <td>
                      <strong>{stage.stage_name}</strong>
                    </td>

                    <td>{stage.unit}</td>

                    <td>
                      {formatNumber(stage.observations)}
                    </td>

                    <td>
                      {formatDecimal(
                        stage.median_duration,
                        2,
                      )}
                    </td>

                    <td>
                      {formatDecimal(
                        stage.average_duration,
                        2,
                      )}
                    </td>

                    <td>
                      {formatDecimal(
                        stage.p90_duration,
                        2,
                      )}
                    </td>

                    <td>
                      {formatDecimal(
                        stage.maximum_duration,
                        2,
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No stage analytics"
            description="Stage analytics data is not currently available."
          />
        )}
      </SectionCard>

      <SectionCard
        title="Stage distribution"
        subtitle="Relative observation volume, not duration contribution"
      >
        {stages.length ? (
          <div className="stage-distribution">
            {stages
              .slice()
              .sort(
                (a, b) =>
                  b.observations - a.observations,
              )
              .map((stage) => {
                const percentage =
                  totalObservations > 0
                    ? (stage.observations /
                        totalObservations) *
                      100
                    : 0

                return (
                  <div
                    className="stage-distribution-row"
                    key={stage.stage_name}
                  >
                    <div className="stage-distribution-label">
                      <strong>{stage.stage_name}</strong>
                      <span>
                        {formatNumber(
                          stage.observations,
                        )}{' '}
                        observations
                      </span>
                    </div>

                    <div className="stage-distribution-track">
                      <span
                        style={{
                          width: `${Math.max(
                            percentage,
                            1,
                          )}%`,
                        }}
                      />
                    </div>

                    <strong>
                      {formatDecimal(percentage, 1)}%
                    </strong>
                  </div>
                )
              })}
          </div>
        ) : (
          <EmptyState
            title="No stage distribution"
            description="There is no stage observation population to visualize."
          />
        )}
      </SectionCard>

      <div className="evidence-banner">
        <div>
          <strong>Important stage interpretation</strong>
          <p>
            The API currently reports seven distinct stage categories.
            <strong> data_loading</strong> and
            <strong> data_loading_1</strong> remain separate because
            their semantics have not been proven equivalent. Staging is
            reported in minutes while the other returned stages are
            reported in seconds.
          </p>
        </div>
      </div>
    </div>
  )
}

function ModelLaboratoryPage({
  model,
}: {
  model: ModelInformationResponse | null
}) {
  return (
    <div className="page-stack">
      <div className="model-hero">
        <div>
          <span className="section-eyebrow">
            Production model
          </span>

          <h2>{model?.model_version ?? 'total-et-v2'}</h2>

          <p>
            ExtraTrees-based total execution model using the engineered
            pre-execution feature set.
          </p>
        </div>

        <div className="model-badge">
          <span />
          DEPLOYED ARTIFACT
        </div>
      </div>

      <div className="metric-grid metric-grid-four">
        <StatCard
          label="Estimator"
          value={
            model?.estimator_type ?? 'ExtraTreesRegressor'
          }
          detail="Production inference estimator"
        />

        <StatCard
          label="Target"
          value={
            model?.target_column ?? 'timetaken_in_sec'
          }
          detail="Execution duration target"
        />

        <StatCard
          label="Target transform"
          value={model?.target_transform ?? 'log1p'}
          detail="Skew-aware target representation"
        />

        <StatCard
          label="Trees"
          value={formatNumber(model?.n_estimators)}
          detail="Configured estimators"
        />
      </div>

      <SectionCard
        title="Model configuration"
        subtitle="Values reported directly by the deployed artifact"
      >
        <div className="detail-grid">
          <div>
            <span>Model version</span>
            <strong>
              {model?.model_version ?? 'total-et-v2'}
            </strong>
          </div>

          <div>
            <span>Estimator</span>
            <strong>
              {model?.estimator_type ??
                'ExtraTreesRegressor'}
            </strong>
          </div>

          <div>
            <span>Target column</span>
            <strong>
              {model?.target_column ??
                'timetaken_in_sec'}
            </strong>
          </div>

          <div>
            <span>Transform</span>
            <strong>
              {model?.target_transform ?? 'log1p'}
            </strong>
          </div>

          <div>
            <span>Min samples leaf</span>
            <strong>
              {formatNumber(model?.min_samples_leaf)}
            </strong>
          </div>

          <div>
            <span>Max features</span>
            <strong>
              {typeof model?.max_features === 'number'
                ? formatDecimal(model.max_features, 2)
                : model?.max_features ?? '—'}
            </strong>
          </div>

          <div>
            <span>Random state</span>
            <strong>
              {formatNumber(model?.random_state)}
            </strong>
          </div>

          <div>
            <span>Artifact format</span>
            <strong>
              {model?.artifact_format ?? 'joblib'}
            </strong>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title="Model comparison"
        subtitle="Production selection context without fabricating unavailable benchmark values"
      >
        <div className="model-comparison">
          <div className="model-row model-row-head">
            <span>Approach</span>
            <span>Role</span>
            <span>Status</span>
          </div>

          <div className="model-row">
            <strong>Global historical median</strong>
            <span>Cold-start fallback</span>
            <span className="status-pill status-neutral">
              PRIMARY FALLBACK
            </span>
          </div>

          <div className="model-row">
            <strong>ExtraTreesRegressor</strong>
            <span>Total execution model</span>
            <span className="status-pill status-success">
              DEPLOYED
            </span>
          </div>

          <div className="model-row">
            <strong>Similar-loader retrieval</strong>
            <span>Cold-start supporting evidence</span>
            <span className="status-pill status-neutral">
              SUPPORTING
            </span>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title="Production rationale"
        subtitle="Why the current architecture is suitable for Modules 1–3"
      >
        <div className="reason-grid">
          <div>
            <span>01</span>
            <strong>Pre-execution boundary</strong>
            <p>
              The deployed model is designed around features available
              before the execution begins.
            </p>
          </div>

          <div>
            <span>02</span>
            <strong>Explicit evidence</strong>
            <p>
              Historical loader behavior and resolved configuration are
              represented as explicit model inputs.
            </p>
          </div>

          <div>
            <span>03</span>
            <strong>Skew-aware target</strong>
            <p>
              The training artifact uses log1p target transformation,
              which is appropriate for the strongly right-skewed duration
              distribution observed in the project data.
            </p>
          </div>

          <div>
            <span>04</span>
            <strong>Honest cold start</strong>
            <p>
              Unseen loaders use a global historical median rather than
              pretending that a loader-specific model has evidence it does
              not have.
            </p>
          </div>
        </div>
      </SectionCard>

      <div className="evidence-banner">
        <div>
          <strong>Training metadata limitation</strong>
          <p>
            The current artifact reports training_rows,
            training_loaders, and training_sprints as zero because those
            metadata values are not stored in the deployed artifact. The
            dashboard intentionally does not substitute guessed values.
          </p>
        </div>
      </div>
    </div>
  )
}

function PredictionStudioPage({
  onPrediction,
  prediction,
}: {
  onPrediction: (
    event: FormEvent<HTMLFormElement>,
    values: {
      loader_name: string
      sprint: string
      ldr_connection_name: string
      datasetname: string
      prevalidation_enabled: boolean | null
      transformation_enabled: boolean | null
    },
  ) => Promise<void>
  prediction: TotalExecutionPredictionResponse | null
}) {
  const [loaderName, setLoaderName] = useState('')
  const [sprint, setSprint] = useState('')
  const [connection, setConnection] = useState('')
  const [dataset, setDataset] = useState('')
  const [prevalidation, setPrevalidation] = useState('unknown')
  const [transformation, setTransformation] = useState('unknown')
  const [submitting, setSubmitting] = useState(false)

  const [formError, setFormError] = useState<string | null>(
    null,
  )

  const handleSubmit = async (
    event: FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault()
    setFormError(null)
    setSubmitting(true)

    try {
      await onPrediction(event, {
        loader_name: loaderName.trim(),
        sprint: sprint.trim(),
        ldr_connection_name: connection.trim(),
        datasetname: dataset.trim(),
        prevalidation_enabled:
          prevalidation === 'unknown'
            ? null
            : prevalidation === 'yes',
        transformation_enabled:
          transformation === 'unknown'
            ? null
            : transformation === 'yes',
      })
    } catch (predictionError) {
      setFormError(
        predictionError instanceof Error
          ? predictionError.message
          : 'Unable to generate prediction.',
      )
    } finally {
      setSubmitting(false)
    }
  }

  const isColdStart =
    prediction?.prediction_source ===
    'GLOBAL_MEDIAN_COLD_START'

  const confidence =
    prediction?.cold_start_confidence ?? null

  const confidenceClass =
    confidence?.level === 'HIGH'
      ? 'status-positive'
      : confidence?.level === 'MEDIUM'
        ? 'status-warning'
        : 'status-neutral'

  const configurationEntries = prediction
    ? Object.entries(prediction.configuration_features)
    : []

  return (
    <div className="prediction-studio">
      <div className="prediction-studio-intro">
        <div>
          <span className="section-eyebrow">
            Prediction Studio
          </span>

          <h2>
            Estimate execution time before the loader starts
          </h2>

          <p>
            Provide the loader configuration available at execution
            trigger time. The prediction service resolves historical
            evidence, configuration features, and the appropriate
            known-loader or cold-start route.
          </p>
        </div>

        <div className="prediction-studio-principle">
          <span className="form-security-dot" />

          <div>
            <strong>Pre-execution only</strong>
            <small>
              Actual execution duration is not supplied to the prediction
              request.
            </small>
          </div>
        </div>
      </div>

      <SectionCard
        title="Loader configuration"
        subtitle="Authoritative identity fields used to resolve configuration"
      >
        <form
          className="prediction-form"
          onSubmit={handleSubmit}
        >
          <div className="form-grid">
            <label>
              <span>Loader name</span>

              <input
                value={loaderName}
                onChange={(event) =>
                  setLoaderName(event.target.value)
                }
                placeholder="e.g. Customer_Load"
                required
              />
            </label>

            <label>
              <span>Sprint</span>

              <input
                value={sprint}
                onChange={(event) =>
                  setSprint(event.target.value)
                }
                placeholder="e.g. 2902_STANDALONE_SPRINT"
                required
              />
            </label>

            <label>
              <span>Loader connection</span>

              <input
                value={connection}
                onChange={(event) =>
                  setConnection(event.target.value)
                }
                placeholder="Connection name"
                required
              />
            </label>

            <label>
              <span>Dataset</span>

              <input
                value={dataset}
                onChange={(event) =>
                  setDataset(event.target.value)
                }
                placeholder="Dataset name"
                required
              />
            </label>

            <label>
              <span>Prevalidation</span>

              <select
                value={prevalidation}
                onChange={(event) =>
                  setPrevalidation(event.target.value)
                }
              >
                <option value="unknown">
                  Not specified
                </option>
                <option value="yes">Enabled</option>
                <option value="no">Disabled</option>
              </select>
            </label>

            <label>
              <span>Transformation</span>

              <select
                value={transformation}
                onChange={(event) =>
                  setTransformation(event.target.value)
                }
              >
                <option value="unknown">
                  Not specified
                </option>
                <option value="yes">Enabled</option>
                <option value="no">Disabled</option>
              </select>
            </label>
          </div>

          {formError && (
            <div className="error-banner">
              <strong>Prediction request rejected</strong>
              <span>{formError}</span>
            </div>
          )}

          <div className="prediction-form-footer">
            <div>
              <span className="form-security-dot" />
              Pre-execution fields only
            </div>

            <button
              type="submit"
              className="primary-button"
              disabled={submitting}
            >
              {submitting
                ? 'Estimating…'
                : 'Generate execution ETA'}

              {!submitting && <span>→</span>}
            </button>
          </div>
        </form>
      </SectionCard>

      {prediction && (
        <>
          <div className="prediction-result prediction-result-enhanced">
            <div className="prediction-result-main">
              <span className="section-eyebrow">
                Estimated total execution
              </span>

              <strong>
                {formatDuration(
                  prediction.predicted_total_seconds,
                )}
              </strong>

              <p>
                {prediction.loader_name} · {prediction.sprint}
              </p>
            </div>

            <div className="prediction-result-source">
              <span>Prediction route</span>

              <strong>
                {isColdStart
                  ? 'Global historical median'
                  : 'Known-loader model'}
              </strong>

              <small>
                {prediction.prediction_source}
              </small>
            </div>

            <div className="prediction-result-source">
              <span>Model</span>

              <strong>{prediction.model_version}</strong>

              <small>
                Requested{' '}
                {formatDateTime(
                  prediction.prediction_timestamp,
                )}
              </small>
            </div>
          </div>

          <div className="metric-grid metric-grid-four">
            <StatCard
              label="Estimated total"
              value={formatDuration(
                prediction.predicted_total_seconds,
              )}
              detail="Pre-execution ETA"
              tone="positive"
            />

            <StatCard
              label="Loader history"
              value={formatNumber(
                prediction.historical_features
                  .loader_prior_execution_count,
              )}
              detail={
                isColdStart
                  ? 'Prior eligible executions'
                  : 'Prior loader executions'
              }
            />

            <StatCard
              label="Configuration coverage"
              value={
                typeof prediction.configuration_features
                  .configuration_coverage_pct === 'number'
                  ? `${formatDecimal(
                      prediction.configuration_features
                        .configuration_coverage_pct,
                      0,
                    )}%`
                  : '—'
              }
              detail="Resolved configuration evidence"
            />

            <StatCard
              label="Cold-start confidence"
              value={
                confidence
                  ? `${confidence.level} · ${formatDecimal(
                      confidence.score * 100,
                      0,
                    )}%`
                  : isColdStart
                    ? 'Unavailable'
                    : 'Known loader'
              }
              detail={
                confidence
                  ? 'Evidence score, not probability'
                  : 'Loader-specific historical route'
              }
              tone={
                confidence?.level === 'HIGH'
                  ? 'positive'
                  : confidence?.level === 'MEDIUM'
                    ? 'warning'
                    : undefined
              }
            />
          </div>

          <div className="dashboard-grid dashboard-grid-main">
            <SectionCard
              title="Prediction evidence"
              subtitle="Why the prediction service selected this route"
            >
              <div className="prediction-evidence-panel">
                <div className="evidence-item">
                  <span className="evidence-icon">01</span>

                  <div>
                    <strong>Prediction source</strong>

                    <p>
                      {isColdStart
                        ? 'No prior eligible execution history was available for this loader, so the global historical median is the primary estimate.'
                        : 'Prior eligible execution history was available, so the known-loader model is the primary prediction route.'}
                    </p>
                  </div>
                </div>

                <div className="evidence-item">
                  <span className="evidence-icon">02</span>

                  <div>
                    <strong>Historical evidence</strong>

                    <p>
                      {formatNumber(
                        prediction.historical_features
                          .loader_prior_execution_count,
                      )}{' '}
                      prior eligible loader executions are visible to
                      the online historical feature provider.
                    </p>
                  </div>
                </div>

                <div className="evidence-item">
                  <span className="evidence-icon">03</span>

                  <div>
                    <strong>Configuration evidence</strong>

                    <p>
                      Configuration coverage is{' '}
                      {typeof prediction.configuration_features
                        .configuration_coverage_pct === 'number'
                        ? `${formatDecimal(
                            prediction.configuration_features
                              .configuration_coverage_pct,
                            0,
                          )}%`
                        : 'not available'}
                      .
                    </p>
                  </div>
                </div>
              </div>
            </SectionCard>

            <SectionCard
              title="Reliability context"
              subtitle="Evidence strength is separated from prediction magnitude"
            >
              {confidence ? (
                <div className="readiness-panel">
                  <div className="readiness-status">
                    <span className="readiness-indicator" />

                    <div>
                      <strong>
                        {confidence.level} confidence evidence
                      </strong>

                      <small>
                        {confidence.reason}
                      </small>
                    </div>
                  </div>

                  <div className="readiness-list">
                    <div>
                      <span>Evidence score</span>
                      <strong>
                        {formatDecimal(
                          confidence.score * 100,
                          1,
                        )}
                        %
                      </strong>
                    </div>

                    <div>
                      <span>Interpretation</span>
                      <strong>
                        Not a calibrated probability
                      </strong>
                    </div>

                    <div>
                      <span>Primary estimate</span>
                      <strong>
                        {formatDuration(
                          prediction.predicted_total_seconds,
                        )}
                      </strong>
                    </div>
                  </div>

                  <span
                    className={`status-pill ${confidenceClass}`}
                  >
                    {confidence.level} evidence
                  </span>
                </div>
              ) : (
                <EmptyState
                  title="Known-loader route"
                  description="This prediction uses historical loader evidence directly. Cold-start evidence scoring only applies when the loader has no prior eligible history."
                />
              )}
            </SectionCard>
          </div>

          {isColdStart && (
            <SectionCard
              title="Similar-loader evidence"
              subtitle="Supporting historical profiles for the unseen loader"
            >
              {prediction.similar_loaders.length ? (
                <div className="table-shell">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Loader</th>
                        <th>Sprint</th>
                        <th>Connection</th>
                        <th>Similarity</th>
                      </tr>
                    </thead>

                    <tbody>
                      {prediction.similar_loaders.map(
                        (loader) => (
                          <tr
                            key={`${loader.loader_name}-${loader.sprint}-${loader.connection}`}
                          >
                            <td>
                              <strong>
                                {loader.loader_name}
                              </strong>
                            </td>

                            <td>{loader.sprint}</td>

                            <td>{loader.connection}</td>

                            <td>
                              <span className="status-pill status-neutral">
                                {formatDecimal(
                                  loader.similarity_score *
                                    100,
                                  1,
                                )}
                                %
                              </span>
                            </td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="No similar loaders retrieved"
                  description="The global fallback remains the primary estimate because no supporting loader profiles were returned."
                />
              )}
            </SectionCard>
          )}

          <SectionCard
            title="Resolved configuration"
            subtitle="Configuration-derived features returned by the prediction service"
          >
            {configurationEntries.length ? (
              <div className="detail-grid">
                {configurationEntries.map(
                  ([key, value]) => (
                    <div key={key}>
                      <span>
                        {key.replaceAll('_', ' ')}
                      </span>

                      <strong>
                        {typeof value === 'number'
                          ? formatDecimal(value, 2)
                          : value}
                      </strong>
                    </div>
                  ),
                )}
              </div>
            ) : (
              <EmptyState
                title="No configuration features returned"
                description="The prediction service did not return configuration-derived features."
              />
            )}
          </SectionCard>

          <SectionCard
            title="Historical context"
            subtitle="Online features calculated strictly before the prediction timestamp"
          >
            <div className="detail-grid">
              <div>
                <span>Global prior executions</span>
                <strong>
                  {formatNumber(
                    prediction.historical_features
                      .global_prior_execution_count,
                  )}
                </strong>
              </div>

              <div>
                <span>Global median</span>
                <strong>
                  {formatDuration(
                    prediction.historical_features
                      .global_prior_median_duration_sec,
                  )}
                </strong>
              </div>

              <div>
                <span>Loader prior executions</span>
                <strong>
                  {formatNumber(
                    prediction.historical_features
                      .loader_prior_execution_count,
                  )}
                </strong>
              </div>

              <div>
                <span>Loader median</span>
                <strong>
                  {formatDuration(
                    prediction.historical_features
                      .loader_prior_median_duration_sec,
                  )}
                </strong>
              </div>

              <div>
                <span>Previous loader duration</span>
                <strong>
                  {formatDuration(
                    prediction.historical_features
                      .loader_previous_duration_sec,
                  )}
                </strong>
              </div>

              <div>
                <span>Time since previous execution</span>
                <strong>
                  {formatDuration(
                    prediction.historical_features
                      .seconds_since_loader_previous_execution,
                  )}
                </strong>
              </div>
            </div>
          </SectionCard>

          <div className="prediction-studio-footnote">
            <span className="form-security-dot" />

            <div>
              <strong>Prediction boundary</strong>

              <p>
                The estimate is generated before execution begins.
                Actual execution duration is retained for later evaluation
                and monitoring rather than being supplied as an input to
                this request.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function MonitoringPage({
  predictions,
}: {
  predictions: PredictionHistoryItem[]
}) {
  const metrics = useMemo(() => {
    const errors = predictions
      .map((item) => item.absolute_error_seconds)
      .filter(
        (value): value is number =>
          value !== null && Number.isFinite(value),
      )

    const relative = predictions
      .map((item) => item.relative_error)
      .filter(
        (value): value is number =>
          value !== null && Number.isFinite(value),
      )

    const within20 = relative.filter(
      (value) => Math.abs(value) <= 0.2,
    )

    const known = predictions.filter((item) => {
      const source =
        item.used_prediction_source?.toLowerCase() ?? ''

      return (
        source.includes('loader') ||
        source.includes('known')
      )
    })

    const unseen = predictions.filter((item) => {
      const source =
        item.used_prediction_source?.toLowerCase() ?? ''

      return (
        source.includes('global') ||
        source.includes('cold') ||
        source.includes('unseen')
      )
    })

    const mae = errors.length
      ? errors.reduce((sum, value) => sum + value, 0) /
        errors.length
      : null

    const median =
      errors.length > 0
        ? (() => {
            const sorted = [...errors].sort(
              (a, b) => a - b,
            )

            const middle = Math.floor(
              sorted.length / 2,
            )

            return sorted.length % 2 === 0
              ? (sorted[middle - 1] +
                  sorted[middle]) /
                  2
              : sorted[middle]
          })()
        : null

    return {
      count: predictions.length,
      observedCount: errors.length,
      mae,
      median,
      coverage:
        relative.length > 0
          ? (within20.length / relative.length) *
            100
          : null,
      knownCount: known.length,
      unseenCount: unseen.length,
    }
  }, [predictions])

  const latest = predictions.slice(0, 10)

  return (
    <div className="page-stack">
      <div className="metric-grid metric-grid-four">
        <StatCard
          label="Predictions"
          value={formatNumber(metrics.count)}
          detail="Persisted prediction records"
        />

        <StatCard
          label="Observed MAE"
          value={formatDuration(metrics.mae)}
          detail={`${formatNumber(
            metrics.observedCount,
          )} with actuals`}
        />

        <StatCard
          label="Median absolute error"
          value={formatDuration(metrics.median)}
          detail="Observed actuals only"
        />

        <StatCard
          label="Within ±20%"
          value={
            metrics.coverage === null
              ? '—'
              : `${formatDecimal(
                  metrics.coverage,
                  1,
                )}%`
          }
          detail="Observed relative-error coverage"
          tone={
            metrics.coverage !== null &&
            metrics.coverage >= 80
              ? 'positive'
              : metrics.coverage !== null &&
                  metrics.coverage >= 60
                ? 'warning'
                : 'default'
          }
        />
      </div>

      <div className="dashboard-grid dashboard-grid-main">
        <SectionCard
          title="Prediction population"
          subtitle="Known-loader versus cold-start routing"
        >
          <div className="operating-picture">
            <div className="operating-primary">
              <span>Known-loader predictions</span>
              <strong>
                {formatNumber(metrics.knownCount)}
              </strong>
              <small>
                Historical loader evidence
              </small>
            </div>

            <div className="operating-stat">
              <span>Cold-start predictions</span>
              <strong>
                {formatNumber(metrics.unseenCount)}
              </strong>
            </div>

            <div className="operating-stat">
              <span>Observed actuals</span>
              <strong>
                {formatNumber(metrics.observedCount)}
              </strong>
            </div>

            <div className="operating-stat">
              <span>Unobserved</span>
              <strong>
                {formatNumber(
                  Math.max(
                    metrics.count -
                      metrics.observedCount,
                    0,
                  ),
                )}
              </strong>
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Reliability interpretation"
          subtitle="Accuracy is evaluated only where actual duration exists"
        >
          <div className="evidence-grid">
            <div className="evidence-item">
              <span className="evidence-icon">01</span>
              <div>
                <strong>MAE</strong>
                <p>
                  Average absolute difference between prediction and
                  observed duration.
                </p>
              </div>
            </div>

            <div className="evidence-item">
              <span className="evidence-icon">02</span>
              <div>
                <strong>±20% coverage</strong>
                <p>
                  Fraction of observed predictions whose relative error is
                  no more than 20%.
                </p>
              </div>
            </div>

            <div className="evidence-item">
              <span className="evidence-icon">03</span>
              <div>
                <strong>Missing actuals</strong>
                <p>
                  Predictions without observed execution duration are not
                  treated as prediction failures.
                </p>
              </div>
            </div>

            <div className="evidence-item">
              <span className="evidence-icon">04</span>
              <div>
                <strong>Risk separation</strong>
                <p>
                  Prediction error and execution risk are separate concepts
                  and are not conflated here.
                </p>
              </div>
            </div>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Prediction feedback"
        subtitle="Most recent persisted prediction records"
      >
        {latest.length ? (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Execution</th>
                  <th>Loader</th>
                  <th>Predicted</th>
                  <th>Actual</th>
                  <th>Error</th>
                  <th>Relative error</th>
                  <th>Route</th>
                  <th>Recorded</th>
                </tr>
              </thead>

              <tbody>
                {latest.map((item) => (
                  <tr key={item.execution_id}>
                    <td>
                      <span className="mono">
                        {item.execution_id}
                      </span>
                    </td>

                    <td>{item.loader_name ?? '—'}</td>

                    <td>
                      {formatDuration(
                        item.predicted_total_seconds,
                      )}
                    </td>

                    <td>
                      {formatDuration(
                        item.actual_total_seconds,
                      )}
                    </td>

                    <td>
                      {formatDuration(
                        item.absolute_error_seconds,
                      )}
                    </td>

                    <td>
                      {item.relative_error === null ||
                      item.relative_error === undefined ||
                      !Number.isFinite(
                        item.relative_error,
                      )
                        ? '—'
                        : `${formatDecimal(
                            item.relative_error * 100,
                            1,
                          )}%`}
                    </td>

                    <td>
                      <span
                        className={`status-pill ${sourceTone(
                          item.used_prediction_source,
                        )}`}
                      >
                        {item.used_prediction_source ??
                          '—'}
                      </span>
                    </td>

                    <td>
                      {formatDateTime(item.recorded_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No prediction feedback yet"
            description="Prediction records will appear here once the system has persisted prediction results."
          />
        )}
      </SectionCard>
    </div>
  )
}

function ColdStartPage({
  loaders,
  predictions,
  onPredict,
}: {
  loaders: LoaderIntelligenceItem[]
  predictions: PredictionHistoryItem[]
  onPredict: () => void
}) {
  const coldStartPredictions = predictions.filter((item) => {
    const source =
      item.used_prediction_source?.toLowerCase() ?? ''

    return (
      source.includes('global') ||
      source.includes('cold') ||
      source.includes('unseen')
    )
  })

  const establishedLoaders = loaders.filter(
    (loader) => loader.execution_count > 0,
  ).length

  const latestColdStart =
    coldStartPredictions[0] ?? null

  return (
    <div className="page-stack">
      <div className="metric-grid metric-grid-four">
        <StatCard
          label="Historical loaders"
          value={formatNumber(estimatedLoaderCount(loaders))}
          detail="Loader profiles currently available"
        />

        <StatCard
          label="Known-loader profiles"
          value={formatNumber(establishedLoaders)}
          detail="Profiles with historical executions"
        />

        <StatCard
          label="Cold-start predictions"
          value={formatNumber(
            coldStartPredictions.length,
          )}
          detail="Persisted global-fallback predictions"
          tone="warning"
        />

        <StatCard
          label="Primary fallback"
          value="Global median"
          detail="Used when loader history is unavailable"
        />
      </div>

      <div className="cold-start-hero">
        <div>
          <span className="section-eyebrow">
            Module 3
          </span>

          <h2>
            Predict responsibly when the loader is new
          </h2>

          <p>
            A new loader does not have enough loader-specific execution
            history to justify pretending that a loader-specific model has
            evidence. The system therefore uses the global historical
            median as the primary estimate and retrieves similar loaders as
            supporting evidence.
          </p>
        </div>

        <button
          className="primary-button"
          onClick={onPredict}
        >
          Test cold-start prediction →
        </button>
      </div>

      <div className="dashboard-grid dashboard-grid-main">
        <SectionCard
          title="Cold-start decision flow"
          subtitle="Current production behavior"
        >
          <div className="architecture-list">
            <div className="architecture-item">
              <span className="architecture-number">01</span>
              <div>
                <strong>Identify loader history</strong>
                <small>
                  Check whether prior eligible executions exist for the
                  requested loader.
                </small>
              </div>
              <span className="architecture-live">
                CHECK
              </span>
            </div>

            <div className="architecture-item">
              <span className="architecture-number">02</span>
              <div>
                <strong>No prior eligible history</strong>
                <small>
                  Do not route an unseen loader through a loader-specific
                  historical model.
                </small>
              </div>
              <span className="architecture-live">
                M3
              </span>
            </div>

            <div className="architecture-item">
              <span className="architecture-number">03</span>
              <div>
                <strong>Global historical median</strong>
                <small>
                  Use the global historical median as the primary
                  prediction fallback.
                </small>
              </div>
              <span className="architecture-live">
                PRIMARY
              </span>
            </div>

            <div className="architecture-item">
              <span className="architecture-number">04</span>
              <div>
                <strong>Similar-loader retrieval</strong>
                <small>
                  Show similar historical profiles as supporting evidence,
                  not as an alternate primary prediction.
                </small>
              </div>
              <span className="architecture-live">
                EVIDENCE
              </span>
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Cold-start safeguards"
          subtitle="What the system deliberately avoids"
        >
          <div className="evidence-grid">
            <div className="evidence-item">
              <span className="evidence-icon">01</span>
              <div>
                <strong>No fabricated history</strong>
                <p>
                  A loader with no eligible execution history is not
                  treated as historically known.
                </p>
              </div>
            </div>

            <div className="evidence-item">
              <span className="evidence-icon">02</span>
              <div>
                <strong>No similarity substitution</strong>
                <p>
                  Similar-loader retrieval provides evidence but does not
                  silently replace the global fallback.
                </p>
              </div>
            </div>

            <div className="evidence-item">
              <span className="evidence-icon">03</span>
              <div>
                <strong>Confidence is evidence strength</strong>
                <p>
                  The cold-start score is not presented as a calibrated
                  probability.
                </p>
              </div>
            </div>

            <div className="evidence-item">
              <span className="evidence-icon">04</span>
              <div>
                <strong>Pre-execution boundary</strong>
                <p>
                  Actual execution duration is reserved for later
                  evaluation.
                </p>
              </div>
            </div>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Latest cold-start feedback"
        subtitle="Persisted predictions using the global fallback route"
      >
        {latestColdStart ? (
          <div className="detail-grid">
            <div>
              <span>Execution</span>
              <strong className="mono">
                {latestColdStart.execution_id}
              </strong>
            </div>

            <div>
              <span>Loader</span>
              <strong>
                {latestColdStart.loader_name ?? '—'}
              </strong>
            </div>

            <div>
              <span>Predicted</span>
              <strong>
                {formatDuration(
                  latestColdStart.predicted_total_seconds,
                )}
              </strong>
            </div>

            <div>
              <span>Actual</span>
              <strong>
                {formatDuration(
                  latestColdStart.actual_total_seconds,
                )}
              </strong>
            </div>

            <div>
              <span>Source</span>
              <strong>
                {latestColdStart.used_prediction_source ??
                  '—'}
              </strong>
            </div>

            <div>
              <span>Recorded</span>
              <strong>
                {formatDateTime(
                  latestColdStart.recorded_at,
                )}
              </strong>
            </div>
          </div>
        ) : (
          <EmptyState
            title="No persisted cold-start feedback"
            description="The backend cold-start route is implemented, but no persisted cold-start prediction feedback is currently available."
          />
        )}
      </SectionCard>
    </div>
  )
}

function DriftMonitoringPage({
  overview,
  loaders,
  stages,
  executions,
}: {
  overview: AnalyticsOverview | null
  loaders: LoaderIntelligenceItem[]
  stages: StageAnalyticsItem[]
  executions: ExecutionHistoryItem[]
}) {
  const successfulRate =
    overview && overview.total_executions > 0
      ? (overview.successful_executions /
          overview.total_executions) *
        100
      : null

  const stageCoverage = stages.reduce(
    (sum, stage) => sum + stage.observations,
    0,
  )

  const loaderExecutionCounts = loaders.map(
    (loader) => loader.execution_count,
  )

  const maxLoaderExecutions =
    loaderExecutionCounts.length
      ? Math.max(...loaderExecutionCounts)
      : null

  const highlyConcentrated =
    maxLoaderExecutions !== null &&
    overview?.total_executions
      ? maxLoaderExecutions /
          overview.total_executions >
        0.25
      : false

  const recentExecutions = executions.slice(0, 20)

  const recentLongExecutions = recentExecutions.filter(
    (execution) =>
      execution.duration_seconds !== null &&
      execution.duration_seconds !== undefined &&
      execution.duration_seconds >= 3600,
  ).length

  const status =
    highlyConcentrated || recentLongExecutions >= 10
      ? 'YELLOW'
      : 'GREEN'

  return (
    <div className="page-stack">
      <div className="metric-grid metric-grid-four">
        <StatCard
          label="Overall status"
          value={status}
          detail="Observable-data screening"
          tone={
            status === 'GREEN'
              ? 'positive'
              : 'warning'
          }
        />

        <StatCard
          label="Execution population"
          value={formatNumber(
            overview?.total_executions,
          )}
          detail="Current historical population"
        />

        <StatCard
          label="Loader population"
          value={formatNumber(loaders.length)}
          detail="Distinct loader profiles"
        />

        <StatCard
          label="Stage observations"
          value={formatNumber(stageCoverage)}
          detail="Returned stage observations"
        />
      </div>

      <div className="drift-status-panel">
        <div className="drift-status-main">
          <span
            className={`drift-status-dot drift-${status.toLowerCase()}`}
          />

          <div>
            <span className="section-eyebrow">
              Current observable signal
            </span>

            <h2>
              {status === 'GREEN'
                ? 'No major observable change signal'
                : 'Moderate observable change detected'}
            </h2>

            <p>
              This Module 1–3 surface is intentionally conservative. It
              reports signals that can be computed from the currently
              available analytics responses rather than claiming a
              statistically validated drift alarm.
            </p>
          </div>
        </div>

        <div className="drift-status-legend">
          <InlineSignal
            label="GREEN"
            value="Stable"
            tone="success"
          />

          <InlineSignal
            label="YELLOW"
            value="Review"
            tone="warning"
          />

          <InlineSignal
            label="RED"
            value="Not inferred"
            tone="neutral"
          />
        </div>
      </div>

      <div className="dashboard-grid dashboard-grid-main">
        <SectionCard
          title="Workload signals"
          subtitle="Observable population characteristics"
        >
          <div className="detail-grid">
            <div>
              <span>Total records</span>
              <strong>
                {formatNumber(overview?.total_records)}
              </strong>
            </div>

            <div>
              <span>Unique loaders</span>
              <strong>
                {formatNumber(overview?.unique_loaders)}
              </strong>
            </div>

            <div>
              <span>Successful rate</span>
              <strong>
                {successfulRate === null
                  ? '—'
                  : `${formatDecimal(
                      successfulRate,
                      1,
                    )}%`}
              </strong>
            </div>

            <div>
              <span>Median duration</span>
              <strong>
                {formatDuration(
                  overview?.median_duration_seconds,
                )}
              </strong>
            </div>

            <div>
              <span>P90 duration</span>
              <strong>
                {formatDuration(
                  overview?.p90_duration_seconds,
                )}
              </strong>
            </div>

            <div>
              <span>Maximum duration</span>
              <strong>
                {formatDuration(
                  overview?.maximum_duration_seconds,
                )}
              </strong>
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Loader distribution signal"
          subtitle="Screening for unusually concentrated historical activity"
        >
          <div className="readiness-panel">
            <div className="readiness-status">
              <span className="readiness-indicator" />

              <div>
                <strong>
                  {highlyConcentrated
                    ? 'One loader dominates the population'
                    : 'No dominant loader signal detected'}
                </strong>

                <small>
                  {maxLoaderExecutions === null
                    ? 'No loader execution distribution is available.'
                    : `Largest loader population: ${formatNumber(
                        maxLoaderExecutions,
                      )} executions.`}
                </small>
              </div>
            </div>

            <div className="readiness-list">
              <div>
                <span>Loader count</span>
                <strong>
                  {formatNumber(loaders.length)}
                </strong>
              </div>

              <div>
                <span>Largest loader count</span>
                <strong>
                  {formatNumber(maxLoaderExecutions)}
                </strong>
              </div>

              <div>
                <span>Recent long executions</span>
                <strong>
                  {formatNumber(
                    recentLongExecutions,
                  )}
                </strong>
              </div>
            </div>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Drift dimensions"
        subtitle="What should be monitored as the project matures"
      >
        <div className="reason-grid">
          <div>
            <span>01</span>
            <strong>Workload distribution</strong>
            <p>
              Track changes in record counts and execution-duration
              distributions over time.
            </p>
          </div>

          <div>
            <span>02</span>
            <strong>Loader composition</strong>
            <p>
              Watch for new loaders, disappearing loaders, and large shifts
              in loader execution frequency.
            </p>
          </div>

          <div>
            <span>03</span>
            <strong>Stage behavior</strong>
            <p>
              Track changes in stage observation volume and duration
              distributions.
            </p>
          </div>

          <div>
            <span>04</span>
            <strong>Feature availability</strong>
            <p>
              Monitor configuration coverage and missingness before using
              those changes as model-drift evidence.
            </p>
          </div>
        </div>
      </SectionCard>

      <div className="evidence-banner">
        <div>
          <strong>Drift boundary</strong>
          <p>
            The current UI provides an operational screening surface.
            Statistically validated drift thresholds, population-stability
            measures, feature-level tests, and automated model retraining
            belong to the later monitoring/retraining modules and are not
            fabricated here.
          </p>
        </div>
      </div>
    </div>
  )
}

function estimatedLoaderCount(
  loaders: LoaderIntelligenceItem[],
): number {
  return loaders.length
}

function App() {
  const [page, setPage] =
    useState<Page>('overview')

  const [overview, setOverview] =
    useState<AnalyticsOverview | null>(null)

  const [executions, setExecutions] =
    useState<ExecutionHistoryItem[]>([])

  const [loaders, setLoaders] =
    useState<LoaderIntelligenceItem[]>([])

  const [stages, setStages] =
    useState<StageAnalyticsItem[]>([])

  const [model, setModel] =
    useState<ModelInformationResponse | null>(null)

  const [predictions, setPredictions] =
    useState<PredictionHistoryItem[]>([])

  const [durationTrend, setDurationTrend] =
    useState<DurationTrendPoint[]>([])

  const [prediction, setPrediction] =
    useState<TotalExecutionPredictionResponse | null>(
      null,
    )

  const [apiStatus, setApiStatus] =
    useState<'checking' | 'online' | 'offline'>(
      'checking',
    )

  const [selectedLoader, setSelectedLoader] =
    useState<string | null>(null)

  const [loading, setLoading] =
    useState(true)

  const [error, setError] =
    useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadDashboard() {
      setLoading(true)
      setError(null)

      try {
        const [
          health,
          overviewData,
          executionData,
          loaderData,
          stageData,
          modelData,
          predictionData,
          durationTrendData,
        ] = await Promise.all([
          getApiHealth(),
          getAnalyticsOverview(),
          getExecutionHistory({
            offset: 0,
            limit: 100,
          }),
          getLoaderIntelligence(),
          getStageAnalytics(),
          getModelInformation(),
          getPredictionHistory({
            offset: 0,
            limit: 100,
          }),
          getDurationTrend(),
        ])

        if (cancelled) {
          return
        }

        setApiStatus(
          health.status === 'ok'
            ? 'online'
            : 'offline',
        )

        setOverview(overviewData)
        setExecutions(executionData.items)
        setLoaders(loaderData.items)
        setStages(stageData.items)
        setModel(modelData)
        setPredictions(predictionData.items)
        setDurationTrend(
          durationTrendData.items,
        )
      } catch (loadError) {
        if (cancelled) {
          return
        }

        setApiStatus('offline')

        setError(
          loadError instanceof Error
            ? loadError.message
            : 'Unable to load dashboard data.',
        )
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadDashboard()

    return () => {
      cancelled = true
    }
  }, [])

  const selectedLoaderProfile = useMemo(
    () =>
      loaders.find(
        (loader) =>
          loader.loader_name ===
          selectedLoader,
      ) ?? null,
    [loaders, selectedLoader],
  )

  async function handlePrediction(
    event: FormEvent<HTMLFormElement>,
    values: {
      loader_name: string
      sprint: string
      ldr_connection_name: string
      datasetname: string
      prevalidation_enabled: boolean | null
      transformation_enabled: boolean | null
    },
  ) {
    event.preventDefault()

    const result =
      await predictTotalExecution({
        ...values,
        prediction_timestamp:
          new Date().toISOString(),
      })

    setPrediction(result)
  }

  function navigate(nextPage: Page) {
    setPage(nextPage)
  }

  const metadata = PAGE_META[page]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">dz</div>

          <div>
            <strong>dataZap</strong>
            <span>Execution Intelligence</span>
          </div>
        </div>

        <div className="environment-chip">
          <span />
          Intelligence environment
        </div>

        <nav className="sidebar-nav">
          {NAVIGATION.map((group) => (
            <div
              className="nav-group"
              key={group.section}
            >
              <div className="nav-section-label">
                {group.section}
              </div>

              {group.items.map((item) => (
                <button
                  key={item.id}
                  className={`nav-item ${
                    page === item.id
                      ? 'nav-item-active'
                      : ''
                  }`}
                  onClick={() =>
                    navigate(item.id)
                  }
                  title={item.description}
                >
                  <span className="nav-icon">
                    {item.icon}
                  </span>

                  <span className="nav-label">
                    {item.label}
                  </span>

                  {item.badge && (
                    <span className="nav-badge">
                      {item.badge}
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="system-version">
            <span>Prediction system</span>
            <strong>Modules 1–3</strong>
          </div>

          <div className="system-version">
            <span>Current model</span>
            <strong>
              {model?.model_version ??
                'total-et-v2'}
            </strong>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>dataZap</span>
            <b>/</b>
            <strong>{metadata.title}</strong>
          </div>

          <div className="topbar-right">
            <div
              className={`api-indicator api-${apiStatus}`}
            >
              <span />

              {apiStatus === 'online'
                ? 'API connected'
                : apiStatus === 'offline'
                  ? 'API unavailable'
                  : 'Checking API'}
            </div>

            <div className="topbar-model">
              <span>MODEL</span>

              <strong>
                {model?.model_version ??
                  'total-et-v2'}
              </strong>
            </div>
          </div>
        </header>

        <div className="content-area">
          <div className="page-header">
            <div>
              <span className="section-eyebrow">
                {metadata.eyebrow}
              </span>

              <h1>{metadata.title}</h1>

              <p>{metadata.description}</p>
            </div>

            <div className="page-header-meta">
              <span>System status</span>

              <strong>
                {apiStatus === 'online'
                  ? 'Operational'
                  : apiStatus === 'offline'
                    ? 'Attention'
                    : 'Checking'}
              </strong>
            </div>
          </div>

          {loading && (
            <div className="loading-bar">
              <span />
            </div>
          )}

          {error && (
            <div className="error-banner">
              <strong>Data connection issue</strong>
              <span>{error}</span>
            </div>
          )}

          {!loading && page === 'overview' && (
            <OverviewPage
              overview={overview}
              executions={executions}
              predictions={predictions}
              durationTrend={durationTrend}
              model={model}
              onNavigate={navigate}
            />
          )}

          {!loading &&
            page === 'loader-intelligence' && (
              <LoaderIntelligencePage
                loaders={loaders}
                onSelect={(loader) => {
                  setSelectedLoader(loader)
                  navigate('loader-detail')
                }}
              />
            )}

          {!loading &&
            page === 'loader-detail' && (
              <LoaderDetailPage
                loader={selectedLoaderProfile}
                stages={stages}
                predictions={predictions}
                model={model}
                onBack={() =>
                  navigate(
                    'loader-intelligence',
                  )
                }
                onPredict={() =>
                  navigate(
                    'prediction-studio',
                  )
                }
                onStageAnalytics={() =>
                  navigate('stage-analytics')
                }
              />
            )}

          {!loading &&
            page === 'execution-explorer' && (
              <ExecutionExplorerPage
                executions={executions}
              />
            )}

          {!loading &&
            page === 'stage-analytics' && (
              <StageAnalyticsPage
                stages={stages}
              />
            )}

          {!loading &&
            page === 'model-laboratory' && (
              <ModelLaboratoryPage
                model={model}
              />
            )}

          {!loading &&
            page === 'prediction-studio' && (
              <PredictionStudioPage
                onPrediction={
                  handlePrediction
                }
                prediction={prediction}
              />
            )}

          {!loading &&
            page === 'prediction-monitoring' && (
              <MonitoringPage
                predictions={predictions}
              />
            )}

          {!loading &&
            page === 'cold-start' && (
              <ColdStartPage
                loaders={loaders}
                predictions={predictions}
                onPredict={() =>
                  navigate(
                    'prediction-studio',
                  )
                }
              />
            )}

          {!loading &&
            page === 'drift-monitoring' && (
              <DriftMonitoringPage
                overview={overview}
                loaders={loaders}
                stages={stages}
                executions={executions}
              />
            )}
        </div>
      </main>
    </div>
  )
}

export default App