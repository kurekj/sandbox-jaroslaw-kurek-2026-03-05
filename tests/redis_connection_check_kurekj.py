import  redis

#REDIS_HOST = "192.168.8.100"
REDIS_HOST = "192.168.23.209"
REDIS_PORT = 6379

try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print("Połączenie z Redisem działa!")

    r.set("test_key", "Hello from Python!")
    print("🔍 test_key =", r.get("test_key"))

except redis.exceptions.ConnectionError as e:
    print("Brak połączenia z Redisem:", e)
