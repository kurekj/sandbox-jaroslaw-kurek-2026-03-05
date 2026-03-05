import pandas as pd

from src.v2.db_utils import execute_query

PROPERTIES_QUERY = """
select
	pp.id as property_id,
    pp.create_date,
    pp.offer_id,
	pp.area,
	pp.price,
	pp.price_m2,
	pp.quarters,
	pp.is_luxury,
	pp.has_attic,
	pp.rooms,
	pp.floors,
	pp.bathrooms,
	pp.flat_type,
	pp.floor,
	pp.kitchen_type,
	pp.plot_area,
	pp.house_type,
	pp.garages,
	pa.type as additional_area_type,
	pa.area as additional_area_area
from
	rds2.properties_property pp
left join rds2.properties_additionalarea pa on
	pp.id = pa.property_id
left join rds2.properties_configuration pc on
	pp.id = pc.property_id
"""

PROPERTIES_DICTS_QUERY = """
select
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__properties_property__quarters) as all_quarters,
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__properties_property__flat_type) as all_flat_type,
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__properties_property__kitchen_type) as all_kitchen_type,
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__properties_property__house_type) as all_house_type,
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__properties_additionalarea__type) as all_additional_area_type
"""

OFFERS_QUERY = """
select
	oo.id as offer_id,
	oo.buildings,
	oo.properties,
	st_x(st_centroid(st_transform(oo.geo_point, 4326))) as lon,
	st_y(st_centroid(st_transform(oo.geo_point, 4326))) as lat,
	oo.type_id,
	oo.facilities,
	oo.natural_sites,
	oo.holiday_location,
	oo.region_id,
	oo.vendor_id,
	rr.country,
    provide_city_name(oo.region_id) as city_name,
    oo.region_id,
	oc.display_type,
    os.display_status
from
	rds2.offers_offer oo
left join rds2.regions_region rr on
	rr.id = oo.region_id
left join rds2.offers_configuration oc on
	oc.offer_id = oo.id
left join rds2.offers_status os on
	os.offer_id = oo.id
"""

OFFERS_DICTS_QUERY = """
select
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__offers_offer__type_id) as all_type_id,
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__offers_offer__facilities) as all_facilities,
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__offers_offer__natural_sites) as all_natural_sites,
	(
	select
		array_agg(id order by id)
	from
		dictionary.rds2__offers_offer__holiday_location) as all_holiday_location
"""

MONTHLY_STATS_QUERY = """
select
	mcts."date",
	mcts.city,
	mcts.offer_type,
	mcts.avg_price_m2,
	mcts.avg_price_m2_studio,
	mcts.avg_price_m2_2_rooms,
	mcts.avg_price_m2_3_rooms,
	mcts.avg_price_m2_4_plus_rooms
from
	stats_index.month_city_type_stats mcts
where
	mcts.complete_data
"""


async def load_offers_data(country: int = 1) -> pd.DataFrame:
    offers = pd.DataFrame(await execute_query(OFFERS_QUERY, (country,)))
    dicts = pd.DataFrame(await execute_query(OFFERS_DICTS_QUERY))

    return offers.merge(dicts, how="cross")


async def load_properties_data() -> pd.DataFrame:
    properties = pd.DataFrame(await execute_query(PROPERTIES_QUERY))
    dicts = pd.DataFrame(await execute_query(PROPERTIES_DICTS_QUERY))

    return properties.merge(dicts, how="cross")


async def load_monthly_stats_data() -> pd.DataFrame:
    return pd.DataFrame(await execute_query(MONTHLY_STATS_QUERY))
