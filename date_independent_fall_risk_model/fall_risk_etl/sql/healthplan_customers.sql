SELECT
	account_id,
	account_number,
	age,
	account_record_type_name,
	acquisition_partner_name
FROM
	DATAHUB.analytics.dim_core__sf_account_customer_details
WHERE
	account_status = 'Active'
	and account_record_type_name = 'Medicaid Customer'
	AND acquisition_partner_name in 
    ('Fidelis Care at Home - NY', 'Florida Community Care', 'United Healthcare Indiana', 'Highmark Health Options Blue Cross Blue Shield Delaware', 'Horizon NJ Health', 'Anthem HealthKeepers Plus')
