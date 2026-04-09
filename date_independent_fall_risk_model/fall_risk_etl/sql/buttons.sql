SELECT 
        account_number,
        CAST([timestamp] AS DATE) AS button_press_date
    FROM datahub.analytics.fct_iot__alarms
    WHERE activator = 'Help Button'
        AND account_number IS NOT NULL