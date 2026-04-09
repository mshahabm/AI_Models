 SELECT 
        account_number,
        cast(event_on AS date) AS step_date,
        daily_steps,
        udi
    FROM datahub.analytics.fct_iot__device_daily_steps
    WHERE account_number IS NOT NULL
        AND daily_steps > 0