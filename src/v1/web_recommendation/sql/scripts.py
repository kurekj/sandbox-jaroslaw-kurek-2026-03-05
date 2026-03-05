VIEWS_AGG_SCRIPT = """
SELECT
    DISTINCT coalesce(nullif(split_part(split_part(ext_cookie, 'algolytics_uuid=', 2),';',1),'')::varchar(36),
    m.algolytics_uuid::varchar(36), v.session_id::varchar(36)) as algolytics_uuid,
    (v.offer_id::numeric::int||'_'||v.property_rooms::numeric::int)::varchar(10) offers
FROM
    web_events.views as v
    LEFT JOIN mapping.uuid_session_map as m ON v.session_id=m.session_id
    JOIN rds2.offers_configuration oc ON v.offer_id::numeric::int = oc.offer_id AND oc.display_type = 1
WHERE
    cast(SUBSTRING(ext_received_dt, '[0-9]+') as bigint) BETWEEN
    cast(EXTRACT(EPOCH FROM timestamp %(from_date)s) as bigint)*1000::bigint AND
    cast(EXTRACT(EPOCH FROM timestamp %(to_date)s) as bigint)*1000::bigint
    AND property_rooms IS NOT NULL
    AND coalesce(nullif(split_part(split_part(ext_cookie, 'algolytics_uuid=', 2),';',1),'')::varchar(36),
    m.algolytics_uuid::varchar(36), v.session_id::varchar(36)) = %(user_uuid)s;"""

APPLICATIONS_AGG_SCRIPT = """
SELECT
    DISTINCT coalesce(nullif(split_part(split_part(ext_cookie, 'algolytics_uuid=', 2),';',1),'')::varchar(36),
    m.algolytics_uuid::varchar(36), v.session_id::varchar(36)) as algolytics_uuid,
    (v.offer_id::numeric::int||'_'||v.property_rooms::numeric::int)::varchar(10) offers
FROM
    web_events.application as v
    LEFT JOIN mapping.uuid_session_map as m ON v.session_id=m.session_id
    JOIN rds2.offers_configuration oc ON v.offer_id::numeric::int = oc.offer_id AND oc.display_type = 1
WHERE
    cast(SUBSTRING(ext_received_dt, '[0-9]+') as bigint) BETWEEN
    cast(EXTRACT(EPOCH FROM timestamp %(from_date)s) as bigint)*1000::bigint AND
    cast(EXTRACT(EPOCH FROM timestamp %(to_date)s) as bigint)*1000::bigint
    AND property_rooms IS NOT NULL
    AND coalesce(nullif(split_part(split_part(ext_cookie, 'algolytics_uuid=', 2),';',1),'')::varchar(36),
    m.algolytics_uuid::varchar(36), v.session_id::varchar(36)) = %(user_uuid)s;"""

VIEWS_GET_DM = """SELECT * FROM agg_tab.web_recommendation_views;"""

APPLICATIONS_GET_DM = """SELECT * FROM agg_tab.web_recommendation_applications;"""

FIND_PROPERTIES = """
SELECT
    DISTINCT ON (offer_id)
    id property_id
FROM
    rds2.properties_property
WHERE
    for_sale
    AND offer_id::text||'_'||rooms::text IN ('%(offer_list)s')
ORDER BY
    offer_id,
    property_id DESC
"""

FIND_USER_UUID = """SELECT uuid FROM km.users_user WHERE id = %(user_id)s"""

FIND_ALGOLYTICS_UUID = """SELECT algolytics_uuid FROM mapping.uuid_session_map WHERE session_id = %(session_id)s"""

GET_DATA_FROM_USER_VECTOR_GH_OFFERS = """
SELECT
    uc.client_id,
    uu.id user_id,
    offer_type_viewed,
    offer_type_leaded,
    rooms_viewed,
    rooms_leaded,
    area_segment_apartments_leaded,
    area_segment_apartments_viewed,
    area_segment_houses_leaded,
    area_segment_houses_viewed,
    price_segment_leaded,
    price_segment_views
FROM
    daily_snapshot.user_vector uv
    LEFT JOIN km.users_user uu ON uv.algolytics_uuid::uuid = uu.uuid
    LEFT JOIN rds2.users_userclient uc ON uu.id = uc.user_id
WHERE
    algolytics_uuid = %(user_id)s
"""

GET_LAST_2_RP_LEADS = """
SELECT
    a.offer_id, max(a.create_date), o.geo_point
FROM
    rds2.applications_application a
    LEFT JOIN rds2.offers_offer o ON a.offer_id = o.id
WHERE
    client_id = %(client_id)s
    AND offer_id IS NOT NULL
GROUP BY
    1, 3
ORDER BY
    2 DESC
LIMIT 2;"""

GET_LAST_2_GH_LEADS = """
SELECT
    rpm.offer_id,
    max(rpm.created_at) create_date,
    coalesce(h.coordinates, ap.coordinates, l.coordinates) geo_point
FROM
    rds2.mdb_applications_application as rpm
    LEFT JOIN km.users_user u ON u.id = rpm.user_id
    LEFT JOIN rds2.mdb_offers_offer o ON o.id = rpm.offer_id
    LEFT JOIN rds2.mdb_houses_house h ON o.house_id = h.id
    LEFT JOIN rds2.mdb_apartments_apartment ap ON o.apartment_id = ap.id
    LEFT JOIN rds2.mdb_lots_lot l ON o.lot_id = l.id
WHERE
    u.id = %(user_id)s
GROUP BY
    1, 3
ORDER BY
    2 DESC
LIMIT 2;"""


FIND_GH_PROPERTIES_ONE_AREA = """
SELECT
    moo.id,
    ST_Distance(
    ST_Transform(oa.coordinates::geometry, 4326)::geography,
    ST_Transform(%(geo_point)s::geometry, 4326)::geography)/1000::float as distance,
    CASE
        WHEN moo.price <= 300000 THEN 150
        WHEN moo.price BETWEEN 300000 AND 400000 THEN 350
        WHEN moo.price BETWEEN 400000 AND 500000 THEN 450
        WHEN moo.price BETWEEN 500000 AND 700000 THEN 600
        WHEN moo.price BETWEEN 700000 AND 1000000 THEN 850
        WHEN moo.price >= 1000000 THEN 1150
    END price_type
FROM
    rds2.mdb_offers_offer moo
    JOIN %(offer_type_tab)s oa ON oa.id = %(id_param)s
    AND ST_Distance(ST_Transform(oa.coordinates::geometry, 4326)::geography,
    ST_Transform(%(geo_point)s::geometry, 4326)::geography)/1000 <= %(distance)s
WHERE
    oa.market_type = 'aftermarket'
    AND moo.is_master
    AND moo.is_active
    AND oa.room_number = %(rooms)s
    AND oa.area BETWEEN %(min_area)s AND %(max_area)s
    AND moo.id NOT IN (%(exclude)s)
"""

FIND_GH_PROPERTIES_TWO_AREAS = (
    """
WITH
total as (
    """
    + FIND_GH_PROPERTIES_ONE_AREA
    + """
    UNION ALL
    """
    + FIND_GH_PROPERTIES_ONE_AREA.replace("geo_point", "geo_point_2")
    + """)

SELECT
    DISTINCT ON (id)
    id,
    distance,
    price_type
FROM
    total
ORDER BY
    1, 3;"""
)


GH_OFFERS_USER_APPLICATION_AGG = """
SELECT
    offer::int
FROM
    agg_tab.web_recommendation_applications
WHERE
    uuid = %(user_id)s """

GH_OFFERS_USER_VIEWS_AGG = """
SELECT
    offer::int
FROM
    agg_tab.web_recommendation_views
WHERE
    uuid = %(user_id)s"""

GH_OFFERS_USER_APPLICATION_LAST = """
SELECT
    replace_offer_uuid_with_numeric(offer_id)::int offer
FROM
    web_events_gh.application
WHERE
    cast(SUBSTRING(ext_received_dt, '[0-9]+') as bigint) >=
    cast(EXTRACT(EPOCH FROM date_trunc('day', current_timestamp)) as bigint)*1000::bigint
    AND offer_id IS NOT NULL
    AND ext_sceuid = %(user_id)s"""

GH_OFFERS_USER_VIEWS = """
SELECT
    replace_offer_uuid_with_numeric(offer_id)::int offer
FROM
    web_events_gh.views
WHERE
    cast(SUBSTRING(ext_received_dt, '[0-9]+') as bigint) >=
    cast(EXTRACT(EPOCH FROM date_trunc('day', current_timestamp)) as bigint)*1000::bigint
    AND offer_id IS NOT NULL
    AND ext_sceuid = %(user_id)s;"""
