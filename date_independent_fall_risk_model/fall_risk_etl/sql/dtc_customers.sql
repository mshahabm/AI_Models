SELECT
	account_number,
	account_id,
	age,
	account_record_type_name,
	account_status
FROM
	datahub.analytics.dim_core__sf_account_customer_details
WHERE
	account_record_type_name = 'Active Customer'
	AND account_status IN ('Active')
	AND brand IN ('Medical Guardian', 'One Call Alert', 'MobileHelp') 