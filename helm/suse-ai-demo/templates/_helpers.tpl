{{/* A release uses the chart appVersion unless an explicit tag or digest is supplied. */}}
{{- define "suse-ai-demo.image" -}}
{{- if .image.digest -}}
{{ .image.repository }}@{{ .image.digest }}
{{- else -}}
{{ .image.repository }}:{{ .image.tag | default .root.Chart.AppVersion }}
{{- end -}}
{{- end -}}

{{- define "suse-ai-demo.otelEnv" -}}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ .root.Values.otel.exporterEndpoint | quote }}
- name: OTEL_SERVICE_NAME
  value: {{ .service | quote }}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_UID
  valueFrom:
    fieldRef:
      fieldPath: metadata.uid
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
- name: OTEL_RESOURCE_ATTRIBUTES
  value: "service.namespace={{ .root.Values.otel.serviceNamespace | default .root.Release.Namespace }},service.version={{ .root.Chart.AppVersion }},deployment.environment.name={{ .root.Values.otel.environment }},service.instance.id=$(POD_UID),suse.ai.managed=true,suse.ai.component.type=application,suse.ai.component.name={{ .service }},k8s.pod.uid=$(POD_UID),k8s.pod.name=$(POD_NAME),k8s.namespace.name=$(POD_NAMESPACE),k8s.node.name=$(NODE_NAME),suse.ai.demo=true"
{{- end -}}

{{- define "suse-ai-demo.apiKeyEnv" -}}
{{- if .config.existingSecret }}
- name: {{ .env }}
  valueFrom:
    secretKeyRef:
      name: {{ .config.existingSecret | quote }}
      key: {{ .config.apiKeyKey | quote }}
{{- end }}
{{- end -}}
