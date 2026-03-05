LEADS_DAYS_AGO_FROM = 30
KNN_THRESHOLD = 0.01
TOP_N_RECOMMENDATION = 6
APPLICATIONS_CHUNK_SIZE = 5000
VIEWS_CHUNK_SIZE = 20000

AREA_SEGMENT_APARTMENTS_DICT = {
    "to_30m2": [0, 30],
    "between_30_and_50m2": [30, 50],
    "between_50_and_70m2": [50, 70],
    "between_70_and_100m2": [70, 100],
    "between_100_and_120m2": [100, 120],
    "above_120m2": [120, 500],
}

AREA_SEGMENT_HOUSES_DICT = {
    "to_100m2": [0, 100],
    "between_100_and_150m2": [100, 150],
    "between_150_and_200m2": [150, 200],
    "between_200_and_250m2": [200, 250],
    "above_250m2": [250, 1000],
}

AREA_SEGMENT = [
    "area_segment_apartments_leaded",
    "area_segment_apartments_viewed",
    "area_segment_houses_leaded",
    "area_segment_houses_viewed",
]

PRICE_SEGMENT = ["price_segment_leaded", "price_segment_views"]

USER_PARAMS = [
    "client_id",
    "user_id",
    "offer_type_viewed",
    "offer_type_leaded",
    "rooms_viewed",
    "rooms_leaded",
    "area_segment_apartments_leaded",
    "area_segment_apartments_viewed",
    "area_segment_houses_leaded",
    "area_segment_houses_viewed",
    "price_segment_leaded",
    "price_segment_views",
]

PRICE_SEGMENT_DICT = {
    "to_300k": 150,
    "between_300k_and_400k": 350,
    "between_400k_and_500k": 450,
    "between_500k_and_700k": 600,
    "between_700k_and_1mln": 850,
    "above_1mln": 1150,
}

NUMBER_OF_GH_OFFERS = 2
