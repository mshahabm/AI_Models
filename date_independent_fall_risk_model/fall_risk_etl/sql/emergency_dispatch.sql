SELECT 
        account_number,
        CAST([timestamp] AS DATE) AS dispatch_date
    FROM datahub.analytics.fct_iot__alarms
    WHERE help_sent = 'Yes' 
        AND type_of_help_sent IN ('Emergency Services', 'Non Medical - Emergency Services')
        AND dispatch_cancelled = 'No'
        AND account_number IS NOT NULL