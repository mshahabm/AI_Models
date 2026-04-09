select
	alarm_id,
	account_number,
	cast(timestamp as date) as alarm_date,
	alarm_category,
	alarm_path,
	alarm_resolution,
	subscriber_reached,
	help_sent,
	type_of_help_sent,
	dispatch_cancelled,
	dispatch_result,
	alarm_summary
from
	datahub.analytics.fct_iot__alarms
where alarm_resolution not in ('FD Abort', 'False Alarm', 'Testing')