select 
    alarm_id,
    account_number,
    cast(timestamp as date) as alarm_date
    from datahub.analytics.fct_iot__alarms 
    where activator = 'Fall Detection'
    and alarm_resolution not in ('Testing', 'Information Request', 'False Alarm', 'FD Abort')