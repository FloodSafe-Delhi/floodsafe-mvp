import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine, text
import os

# Uses DATABASE_URL from env, default to local postgres used in docker-compose
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/floodsafe")
engine = create_engine(DATABASE_URL)

def load_sensor_df(hotspot_id):
    q = text("""
      SELECT ts, water_level FROM sensor_readings
      WHERE hotspot_id = :hs AND water_level IS NOT NULL
      AND ts >= now() - interval '14 days'
      ORDER BY ts;
    """)
    df = pd.read_sql(q, engine, params={"hs": hotspot_id})
    if df.empty:
        return None
    df = df.rename(columns={"ts":"ds","water_level":"y"})
    df['ds'] = pd.to_datetime(df['ds'])
    # resample hourly and interpolate so Prophet has regular intervals
    df = df.set_index('ds').resample('H').mean().interpolate().reset_index()
    return df

def train_and_forecast(hotspot_id, hours=24):
    df = load_sensor_df(hotspot_id)
    if df is None:
        print("no data for", hotspot_id)
        return
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=hours, freq='H')
    forecast = m.predict(future)
    preds = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(hours)
    # write preds to DB (simple inserts; dedupe if you run multiple times)
    with engine.begin() as conn:
        for _, row in preds.iterrows():
            conn.execute(text("""
              INSERT INTO predictions (hotspot_id, ds, yhat, yhat_lower, yhat_upper)
              VALUES (:hs, :ds, :yhat, :l, :u)
            """), {"hs": hotspot_id, "ds": row['ds'], "yhat": float(row['yhat']), "l": float(row['yhat_lower']), "u": float(row['yhat_upper'])})
    print("wrote preds for", hotspot_id)

if __name__ == "__main__":
    # For now, use a static list — later you can query distinct hotspots from DB
    hotspots = ["KarolBagh","Okhla"]
    for h in hotspots:
        train_and_forecast(h)
