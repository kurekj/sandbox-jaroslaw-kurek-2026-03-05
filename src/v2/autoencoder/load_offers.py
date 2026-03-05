import pandas as pd

from src.v2.db_utils import execute_query

OFFER_QUERY = """
select
	oo.id,
	oo.create_date,
	oo.buildings,
	oo.properties,
	st_x(st_centroid(st_transform(oo.geo_point, 4326))) as lon,
	st_y(st_centroid(st_transform(oo.geo_point, 4326))) as lat,
	lower(oo.construction_date_range) as construction_date_start,
	upper(oo.construction_date_range) as construction_date_end,
	oo.type_id,
	(
	select
		array_agg(id)
	from
		dictionary.rds2__offers_offer__type_id) as all_type_id,
	oo.facilities,
	(
	select
		array_agg(id)
	from
		dictionary.rds2__offers_offer__facilities) as all_facilities,
	oo.natural_sites,
	(
	select
		array_agg(id)
	from
		dictionary.rds2__offers_offer__natural_sites roons) as all_natural_sites,
	oo.holiday_location,
	(
	select
		array_agg(id)
	from
		dictionary.rds2__offers_offer__holiday_location) as all_holiday_location,
	oo.region_id,
	oo.vendor_id,
	rr.country,
	os.additional_areas,
	op.poi
from
	rds2.offers_offer oo
left join rds2.regions_region rr on
	rr.id = oo.region_id
left join rds2.offers_stats os on
	os.offer_id = oo.id
left join rds2.offers_configuration oc on
	oc.offer_id = oo.id
left join daily_snapshot.offers_poi op on
	op.offer_id = oo.id
where
	oo.vendor_id not in (979, 11130, 4879)
	and not oc.is_invalid
    and op.poi is not null
order by
	create_date desc
"""


async def load_offers_data() -> pd.DataFrame:
    leads_data = await execute_query(OFFER_QUERY)
    return pd.DataFrame(leads_data)
