{{/*
Expand the name of the chart.
*/}}
{{- define "async-toolformer-orchestrator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "async-toolformer-orchestrator.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "async-toolformer-orchestrator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "async-toolformer-orchestrator.labels" -}}
helm.sh/chart: {{ include "async-toolformer-orchestrator.chart" . }}
{{ include "async-toolformer-orchestrator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "async-toolformer-orchestrator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "async-toolformer-orchestrator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "async-toolformer-orchestrator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "async-toolformer-orchestrator.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the Redis URL
*/}}
{{- define "async-toolformer-orchestrator.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- if .Values.redis.auth.enabled }}
redis://:{{ .Values.redis.auth.password }}@{{ include "async-toolformer-orchestrator.fullname" . }}-redis-master:6379
{{- else }}
redis://{{ include "async-toolformer-orchestrator.fullname" . }}-redis-master:6379
{{- end }}
{{- else }}
{{- .Values.externalRedis.url }}
{{- end }}
{{- end }}

{{/*
Create Prometheus metrics labels
*/}}
{{- define "async-toolformer-orchestrator.metricsLabels" -}}
app.kubernetes.io/name: {{ include "async-toolformer-orchestrator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: metrics
{{- end }}

{{/*
Create NetworkPolicy name
*/}}
{{- define "async-toolformer-orchestrator.networkPolicyName" -}}
{{ include "async-toolformer-orchestrator.fullname" . }}-network-policy
{{- end }}

{{/*
Create PodDisruptionBudget name
*/}}
{{- define "async-toolformer-orchestrator.pdbName" -}}
{{ include "async-toolformer-orchestrator.fullname" . }}-pdb
{{- end }}

{{/*
Create HorizontalPodAutoscaler name
*/}}
{{- define "async-toolformer-orchestrator.hpaName" -}}
{{ include "async-toolformer-orchestrator.fullname" . }}-hpa
{{- end }}

{{/*
Create VerticalPodAutoscaler name
*/}}
{{- define "async-toolformer-orchestrator.vpaName" -}}
{{ include "async-toolformer-orchestrator.fullname" . }}-vpa
{{- end }}

{{/*
Create ServiceMonitor name
*/}}
{{- define "async-toolformer-orchestrator.serviceMonitorName" -}}
{{ include "async-toolformer-orchestrator.fullname" . }}-metrics
{{- end }}

{{/*
Create PrometheusRule name
*/}}
{{- define "async-toolformer-orchestrator.prometheusRuleName" -}}
{{ include "async-toolformer-orchestrator.fullname" . }}-alerts
{{- end }}

{{/*
Common annotations
*/}}
{{- define "async-toolformer-orchestrator.annotations" -}}
app.kubernetes.io/name: {{ include "async-toolformer-orchestrator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ include "async-toolformer-orchestrator.chart" . }}
{{- end }}

{{/*
Create security context for init containers
*/}}
{{- define "async-toolformer-orchestrator.initSecurityContext" -}}
runAsNonRoot: true
runAsUser: 65534
runAsGroup: 65534
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop:
    - ALL
{{- end }}

{{/*
Create volume mounts for configuration
*/}}
{{- define "async-toolformer-orchestrator.configVolumeMounts" -}}
- name: config
  mountPath: /app/config
  readOnly: true
{{- if .Values.persistence.enabled }}
- name: data
  mountPath: /app/data
{{- end }}
- name: tmp
  mountPath: /tmp
{{- end }}

{{/*
Create volumes for configuration
*/}}
{{- define "async-toolformer-orchestrator.configVolumes" -}}
- name: config
  configMap:
    name: {{ include "async-toolformer-orchestrator.fullname" . }}-config
{{- if .Values.persistence.enabled }}
- name: data
  persistentVolumeClaim:
    claimName: {{ include "async-toolformer-orchestrator.fullname" . }}-data
{{- end }}
- name: tmp
  emptyDir: {}
{{- end }}

{{/*
Generate environment variables from values
*/}}
{{- define "async-toolformer-orchestrator.envVars" -}}
{{- range $key, $value := .Values.config.orchestrator }}
- name: ORCHESTRATOR_{{ $key | upper | replace "." "_" }}
  value: {{ $value | quote }}
{{- end }}
{{- range $key, $value := .Values.config.rateLimiting }}
- name: RATE_LIMITING_{{ $key | upper | replace "." "_" }}
  value: {{ $value | quote }}
{{- end }}
{{- range $key, $value := .Values.config.observability }}
- name: OBSERVABILITY_{{ $key | upper | replace "." "_" }}
  value: {{ $value | quote }}
{{- end }}
{{- range $key, $value := .Values.config.security }}
- name: SECURITY_{{ $key | upper | replace "." "_" }}
  value: {{ $value | quote }}
{{- end }}
{{- if .Values.redis.enabled }}
- name: REDIS_URL
  value: {{ include "async-toolformer-orchestrator.redisUrl" . }}
{{- end }}
{{- end }}