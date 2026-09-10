import type { DurationTrendPoint } from '../../api'
import './DurationTrendChart.css'

type DurationTrendChartProps = {
  points: DurationTrendPoint[]
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return '—'
  }

  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`
  }

  const minutes = seconds / 60

  if (minutes < 60) {
    return `${minutes.toFixed(1)}m`
  }

  return `${(minutes / 60).toFixed(1)}h`
}

function getMaxValue(points: DurationTrendPoint[]): number {
  const values = points
    .flatMap((point) => [
      point.average_duration_seconds ?? 0,
      point.median_duration_seconds ?? 0,
    ])
    .filter((value) => Number.isFinite(value))

  return Math.max(...values, 1)
}

function getPointCoordinates(
  points: DurationTrendPoint[],
  valueSelector: (
    point: DurationTrendPoint,
  ) => number | null,
  maxValue: number,
): string {
  return points
    .map((point, index) => {
      const x =
        points.length === 1
          ? 500
          : (index / (points.length - 1)) * 1000

      const value = valueSelector(point) ?? 0

      const y = Math.max(
        20,
        260 - (value / maxValue) * 230,
      )

      return `${x},${y}`
    })
    .join(' ')
}

export function DurationTrendChart({
  points,
}: DurationTrendChartProps) {
  if (points.length === 0) {
    return (
      <div className="chart-empty">
        <strong>No duration trend data</strong>
        <span>
          Historical execution data is not available yet.
        </span>
      </div>
    )
  }

  const maxValue = getMaxValue(points)
  const visiblePoints = points.slice(-30)

  const averagePoints = getPointCoordinates(
    visiblePoints,
    (point) => point.average_duration_seconds,
    maxValue,
  )

  const medianPoints = getPointCoordinates(
    visiblePoints,
    (point) => point.median_duration_seconds,
    maxValue,
  )

  return (
    <div className="trend-chart">
      <div className="trend-chart-header">
        <div className="trend-chart-legend">
          <span className="trend-legend-item">
            <span className="trend-legend-line trend-legend-average" />
            Average
          </span>

          <span className="trend-legend-item">
            <span className="trend-legend-line trend-legend-median" />
            Median
          </span>
        </div>

        <span className="trend-chart-context">
          Latest {visiblePoints.length} execution dates
        </span>
      </div>

      <div className="trend-chart-body">
        <div className="trend-chart-y-axis">
          <span>{formatDuration(maxValue)}</span>
          <span>{formatDuration(maxValue / 2)}</span>
          <span>0s</span>
        </div>

        <div className="trend-chart-area">
          <div className="trend-grid-line trend-grid-top" />
          <div className="trend-grid-line trend-grid-middle" />
          <div className="trend-grid-line trend-grid-bottom" />

          <svg
            className="trend-svg"
            viewBox="0 0 1000 280"
            preserveAspectRatio="none"
            role="img"
            aria-label="Average and median execution duration trend"
          >
            <polyline
              points={averagePoints}
              fill="none"
              className="trend-line trend-line-average"
              strokeWidth="3"
              vectorEffect="non-scaling-stroke"
            />

            <polyline
              points={medianPoints}
              fill="none"
              className="trend-line trend-line-median"
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />

            {visiblePoints.map((point, index) => {
              const x =
                visiblePoints.length === 1
                  ? 500
                  : (index /
                      (visiblePoints.length - 1)) *
                    1000

              const averageValue =
                point.average_duration_seconds

              const medianValue =
                point.median_duration_seconds

              const averageY =
                averageValue === null ||
                !Number.isFinite(averageValue)
                  ? null
                  : Math.max(
                      20,
                      260 -
                        (averageValue / maxValue) *
                          230,
                    )

              const medianY =
                medianValue === null ||
                !Number.isFinite(medianValue)
                  ? null
                  : Math.max(
                      20,
                      260 -
                        (medianValue / maxValue) *
                          230,
                    )

              return (
                <g
                  key={`${point.execution_date}-${index}`}
                >
                  {averageY !== null && (
                    <circle
                      className="trend-point trend-point-average"
                      cx={x}
                      cy={averageY}
                      r="4"
                    >
                      <title>
                        {point.execution_date} · Average:{' '}
                        {formatDuration(averageValue)}
                      </title>
                    </circle>
                  )}

                  {medianY !== null && (
                    <circle
                      className="trend-point trend-point-median"
                      cx={x}
                      cy={medianY}
                      r="3"
                    >
                      <title>
                        {point.execution_date} · Median:{' '}
                        {formatDuration(medianValue)}
                      </title>
                    </circle>
                  )}
                </g>
              )
            })}
          </svg>

          <div className="trend-labels">
            {visiblePoints.map((point, index) => {
              const interval = Math.max(
                1,
                Math.ceil(visiblePoints.length / 6),
              )

              const shouldShow =
                visiblePoints.length <= 10 ||
                index === 0 ||
                index === visiblePoints.length - 1 ||
                index % interval === 0

              return (
                <span
                  key={`${point.execution_date}-label`}
                  className={
                    shouldShow
                      ? ''
                      : 'trend-label-hidden'
                  }
                >
                  {point.execution_date}
                </span>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

