const API_PREFIX = '/api'

export type HistoricalFeatures = {
  global_prior_execution_count: number
  global_prior_mean_duration_sec: number | null
  global_prior_median_duration_sec: number | null
  global_prior_std_duration_sec: number | null
  loader_prior_execution_count: number
  loader_prior_mean_duration_sec: number | null
  loader_prior_median_duration_sec: number | null
  loader_prior_std_duration_sec: number | null
  loader_previous_duration_sec: number | null
  seconds_since_loader_previous_execution: number | null
}

export type TotalExecutionPredictionRequest = {
  loader_name: string
  sprint: string
  ldr_connection_name: string
  datasetname: string
  prevalidation_enabled: boolean | null
  transformation_enabled: boolean | null
  prediction_timestamp?: string
}

export type ColdStartConfidence = {
  level: 'LOW' | 'MEDIUM' | 'HIGH'
  score: number
  reason: string
}

export type SimilarLoader = {
  loader_name: string
  sprint: string
  connection: string
  similarity_score: number
}

export type TotalExecutionPredictionResponse = {
  predicted_total_seconds: number
  prediction_timestamp: string
  loader_name: string
  sprint: string
  model_version: string
  prediction_source: string
  historical_features: HistoricalFeatures
  configuration_features: Record<string, number | string>
  cold_start_confidence: ColdStartConfidence | null
  similar_loaders: SimilarLoader[]
}

export type AnalyticsOverview = {
  total_executions: number
  successful_executions: number
  stopped_executions: number
  other_executions: number
  valid_training_targets: number
  unique_loaders: number
  total_records: number
  average_duration_seconds: number | null
  median_duration_seconds: number | null
  p90_duration_seconds: number | null
  maximum_duration_seconds: number | null
  earliest_execution: string | null
  latest_execution: string | null
}

export type ExecutionHistoryItem = {
  execution_id: string
  loader_name: string | null
  status: string | null
  sprint: string | null
  start_time: string | null
  end_time: string | null
  duration_seconds: number | null
  dataset_count: number
  total_records: number
  success_records: number
  error_records: number
  prevalidation_duration_seconds: number | null
  transformation_duration_seconds: number | null
  staging_duration_minutes: number | null
  data_loading_duration_seconds: number | null
  datamart_approval_duration_seconds: number | null
  dataloading_approval_duration_seconds: number | null
}

export type ExecutionHistoryResponse = {
  items: ExecutionHistoryItem[]
  total: number
  offset: number
  limit: number
}

export type DurationTrendPoint = {
  execution_date: string
  execution_count: number
  average_duration_seconds: number | null
  median_duration_seconds: number | null
}

export type DurationTrendResponse = {
  items: DurationTrendPoint[]
}

export type LoaderIntelligenceItem = {
  loader_name: string
  execution_count: number
  successful_execution_count: number
  average_duration_seconds: number | null
  median_duration_seconds: number | null
  p90_duration_seconds: number | null
  maximum_duration_seconds: number | null
  total_records: number
  latest_execution: string | null
}

export type LoaderIntelligenceResponse = {
  items: LoaderIntelligenceItem[]
}

export type StageAnalyticsItem = {
  stage_name: string
  unit: 'seconds' | 'minutes'
  observations: number
  average_duration: number | null
  median_duration: number | null
  p90_duration: number | null
  maximum_duration: number | null
}

export type StageAnalyticsResponse = {
  items: StageAnalyticsItem[]
}

export type ModelInformationResponse = {
  model_version: string
  estimator_type: string
  target_column: string
  target_transform: string
  training_rows: number
  training_loaders: number
  training_sprints: number
  n_estimators: number
  min_samples_leaf: number
  max_features: number | string | null
  random_state: number
  artifact_format: string
}

export type PredictionHistoryItem = {
  execution_id: string
  loader_name: string | null
  sprint: string | null
  prediction_timestamp: string | null
  predicted_total_seconds: number | null
  actual_total_seconds: number | null
  prediction_error_seconds: number | null
  absolute_error_seconds: number | null
  relative_error: number | null
  execution_status: string | null
  used_prediction_source: string | null
  reliability_score: number | null
  model_version: string | null
  training_eligible: boolean
  recorded_at: string | null
}

export type PredictionHistoryResponse = {
  items: PredictionHistoryItem[]
  total: number
  offset: number
  limit: number
}

export type ApiError = {
  detail?: string
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    ...options,
    headers: {
      Accept: 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    let message = `API request failed with status ${response.status}.`

    try {
      const errorBody = (await response.json()) as ApiError

      if (errorBody.detail) {
        message = errorBody.detail
      }
    } catch {
      // Keep the HTTP-status fallback message.
    }

    throw new Error(message)
  }

  return (await response.json()) as T
}

export async function getAnalyticsOverview(): Promise<AnalyticsOverview> {
  return request<AnalyticsOverview>('/analytics/overview')
}

export async function getExecutionHistory(
  params: {
    offset?: number
    limit?: number
    loaderName?: string
    status?: string
  } = {},
): Promise<ExecutionHistoryResponse> {
  const searchParams = new URLSearchParams()

  if (params.offset !== undefined) {
    searchParams.set('offset', String(params.offset))
  }

  if (params.limit !== undefined) {
    searchParams.set('limit', String(params.limit))
  }

  if (params.loaderName) {
    searchParams.set('loader_name', params.loaderName)
  }

  if (params.status) {
    searchParams.set('status', params.status)
  }

  const query = searchParams.toString()

  return request<ExecutionHistoryResponse>(
    `/analytics/executions${query ? `?${query}` : ''}`,
  )
}

export async function getDurationTrend(): Promise<DurationTrendResponse> {
  return request<DurationTrendResponse>('/analytics/duration-trend')
}

export async function getLoaderIntelligence(): Promise<LoaderIntelligenceResponse> {
  return request<LoaderIntelligenceResponse>('/analytics/loaders')
}

export async function getStageAnalytics(): Promise<StageAnalyticsResponse> {
  return request<StageAnalyticsResponse>('/analytics/stages')
}

export async function getModelInformation(): Promise<ModelInformationResponse> {
  return request<ModelInformationResponse>('/analytics/model')
}

export async function getPredictionHistory(
  params: {
    offset?: number
    limit?: number
  } = {},
): Promise<PredictionHistoryResponse> {
  const searchParams = new URLSearchParams()

  if (params.offset !== undefined) {
    searchParams.set('offset', String(params.offset))
  }

  if (params.limit !== undefined) {
    searchParams.set('limit', String(params.limit))
  }

  const query = searchParams.toString()

  return request<PredictionHistoryResponse>(
    `/analytics/predictions${query ? `?${query}` : ''}`,
  )
}

export async function predictTotalExecution(
  requestBody: TotalExecutionPredictionRequest,
): Promise<TotalExecutionPredictionResponse> {
  return request<TotalExecutionPredictionResponse>('/predict/total', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  })
}

export async function getApiHealth(): Promise<{
  status: string
  service: string
  environment: string
}> {
  return request('/health')
}