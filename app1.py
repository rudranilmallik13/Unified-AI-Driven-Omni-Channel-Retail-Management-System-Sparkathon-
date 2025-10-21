import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import pydeck as pdk
from datetime import datetime
import os

# === Load Data ===
customers   = pd.read_csv("data/customers.csv")
stores      = pd.read_csv("data/stores.csv")
warehouses  = pd.read_csv("data/warehouses.csv")
inventory   = pd.read_csv("data/inventory.csv")
products    = pd.read_csv("data/products.csv")
orders      = pd.read_csv("data/orders.csv")

locations = pd.concat([
    customers.assign(type="customer"),
    stores.assign(type="store"),
    warehouses.assign(type="warehouse")
])[["id", "name", "lat", "lon", "type"]]

# === Create log file if missing ===
log_path = "logs/fulfillment_log.csv"
os.makedirs("logs", exist_ok=True)
if not os.path.exists(log_path):
    pd.DataFrame(columns=[
        "timestamp","order_id","product_id",
        "route","distance_km","cost_rs","time_hr"
    ]).to_csv(log_path, index=False)

# === Helper Functions ===
def get_location(loc_id):
    row = locations[locations['id'] == loc_id].iloc[0]
    return (row['lat'], row['lon'])

def find_nearest_store_with_stock(customer_id, product_id):
    customer_loc = get_location(customer_id)
    candidates = []
    for _, store in stores.iterrows():
        stock = inventory[
            (inventory['location_id'] == store['id']) &
            (inventory['product_id'] == product_id)
        ]['stock']
        if not stock.empty and stock.values[0] > 0:
            dist = geodesic(customer_loc, (store['lat'], store['lon'])).km
            candidates.append((store['id'], dist))
    return sorted(candidates, key=lambda x: x[1])[0][0] if candidates else None

def fallback_route(customer_id, product_id):
    customer_loc = get_location(customer_id)
    best, best_dist = None, float("inf")
    for _, wh in warehouses.iterrows():
        stock = inventory[
            (inventory['location_id'] == wh['id']) &
            (inventory['product_id'] == product_id)
        ]['stock']
        if stock.empty or stock.values[0] == 0:
            continue
        for _, store in stores.iterrows():
            dist = (
                geodesic((wh['lat'], wh['lon']), (store['lat'], store['lon'])).km +
                geodesic((store['lat'], store['lon']), customer_loc).km
            )
            if dist < best_dist:
                best_dist = dist
                best = (wh['id'], store['id'])
    return best

def route_order(customer_id, product_id):
    steps, note = [], ""
    store_id = find_nearest_store_with_stock(customer_id, product_id)
    if store_id:
        steps = [("Store", store_id), ("Customer", customer_id)]
        note  = "Store Fulfilled"
    else:
        fallback = fallback_route(customer_id, product_id)
        if fallback:
            wh_id, st_id = fallback
            steps = [("Warehouse", wh_id), ("Store", st_id), ("Customer", customer_id)]
            note  = "Warehouse → Store → Customer"
        else:
            note  = "❌ Out of Stock Everywhere"
    return steps, note

# === Streamlit Config ===
st.set_page_config(page_title="Walmart Unified Dashboard", layout="wide")
st.title("🏬 Walmart Unified Retail Dashboard")

# === Tabs ===
tab1, tab2 = st.tabs(["📦 Fulfillment Optimizer", "📈 Forecasting Dashboard"])

# --------------------------------------------------------------------------
# TAB 1  –  Fulfillment Optimizer  (unchanged)
# --------------------------------------------------------------------------
with tab1:
    order_id = st.selectbox("Select Order ID", orders['order_id'].unique())
    if order_id:
        order          = orders[orders['order_id'] == order_id].iloc[0]
        customer_id    = order['customer_id']
        product_id     = order['product_id']
        quantity       = order['quantity']
        product_name   = products.loc[products['id'] == product_id, 'name'].iloc[0]
        customer_name  = customers.loc[customers['id'] == customer_id, 'name'].iloc[0]

        simulate_stockout = st.checkbox("🧪 Simulate Store Stockout")
        if simulate_stockout:
            inventory.loc[
                (inventory['location_id'].isin(stores['id'])) &
                (inventory['product_id'] == product_id),
                'stock'
            ] = 0

        st.subheader(f"🧾 Order {order_id} Details")
        st.markdown(f"""
        - 👤 Customer: **{customer_name}** (`{customer_id}`)
        - 📦 Product: **{product_name}** (`{product_id}`)
        - 🔢 Quantity: `{quantity}`
        """)

        route, note = route_order(customer_id, product_id)

        if route:
            st.success("✅ Fulfillment Route: " +
                       " → ".join([f"{s[0]} ({s[1]})" for s in route]))
            st.info(f"📦 Fulfillment Type: **{note}**")

            cost_km, wh_speed, st_speed = 10, 30, 40
            total_dist = est_time = cost = 0

            for i in range(len(route) - 1):
                loc1 = get_location(route[i][1])
                loc2 = get_location(route[i+1][1])
                d    = geodesic(loc1, loc2).km
                total_dist += d
                cost       += d * cost_km
                est_time   += d / (wh_speed if route[i][0]=="Warehouse" else st_speed)

            st.write(f"🚗 **Distance**: `{total_dist:.2f} km`")
            st.write(f"⏱️ **Delivery Time**: `{est_time:.1f} hrs`")
            st.write(f"💰 **Cost**: ₹`{cost:.2f}`")

            # Log once per order
            log_df = pd.read_csv(log_path)
            if order_id not in log_df["order_id"].values:
                pd.DataFrame([{
                    "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_id"   : order_id,
                    "product_id" : product_id,
                    "route"      : " → ".join([f"{x[0]}({x[1]})" for x in route]),
                    "distance_km": round(total_dist, 2),
                    "cost_rs"    : round(cost, 2),
                    "time_hr"    : round(est_time, 2)
                }]).to_csv(log_path, mode="a", header=False, index=False)

            # Deduct stock from first source
            inventory.loc[
                (inventory['location_id'] == route[0][1]) &
                (inventory['product_id'] == product_id),
                'stock'
            ] -= quantity

            # Map
            df_map = pd.DataFrame(
                [get_location(s[1]) for s in route],
                columns=["lat","lon"]
            ).assign(step=[f"{s[0]}: {s[1]}" for s in route])

            line_df = pd.DataFrame([{
                "from_lon": df_map.loc[i,"lon"], "from_lat": df_map.loc[i,"lat"],
                "to_lon"  : df_map.loc[i+1,"lon"], "to_lat": df_map.loc[i+1,"lat"]
            } for i in range(len(df_map)-1)])

            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude = df_map["lat"].mean(),
                    longitude= df_map["lon"].mean(),
                    zoom     = 5.5),
                layers=[
                    pdk.Layer("LineLayer",     data=line_df,
                              get_source_position='[from_lon, from_lat]',
                              get_target_position='[to_lon, to_lat]',
                              get_width=4, get_color=[255,100,0]),
                    pdk.Layer("ScatterplotLayer", data=df_map,
                              get_position='[lon, lat]',
                              get_color='[0,100,200]', get_radius=20000)
                ],
                tooltip={"text": "{step}"}
            ))
        else:
            st.error("❌ Cannot fulfill this order from any source.")

        # Sidebar export
        with st.sidebar:
            st.header("📤 Export Logs")
            if os.path.exists(log_path):
                with open(log_path, "rb") as f:
                    st.download_button("Download Fulfillment Logs", f,
                                       file_name="fulfillment_log.csv")

# --------------------------------------------------------------------------
# TAB 2  –  NEW Unified AI‑Driven Retail Forecasting Dashboard
# --------------------------------------------------------------------------
with tab2:
    # The overall app title is already set; add a header for this tab
    st.header("📊 Unified AI‑Driven Retail Forecasting Dashboard")

    # -----------------------------
    # Simulate Unified Dataset
    # -----------------------------
    @st.cache_data
    def generate_unified_data():
        np.random.seed(42)
        dates    = pd.date_range(start='2025-06-20', periods=21)
        regions  = ['North','South','East','West']
        products = ['Product A','Product B','Product C']
        channels = ['Online','Store','Marketplace','3rdParty']
        data = []
        for region in regions:
            for product in products:
                for date in dates:
                    for ch in channels:
                        lam = {'Online':100,'Store':80,'Marketplace':60,'3rdParty':40}[ch]
                        demand = np.random.poisson(lam=lam) + (0 if ch!='3rdParty' else np.random.randint(-5,10))
                        if date == pd.Timestamp('2025-07-04'):
                            demand += 30  # holiday spike
                        data.append({
                            'Date':date,'Region':region,'Product':product,
                            'Channel':ch,'Forecasted_Demand':max(0,demand)
                        })
        return pd.DataFrame(data)

    df = generate_unified_data()

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    product  = st.sidebar.selectbox("Select Product", df["Product"].unique())
    region   = st.sidebar.selectbox("Select Region",  df["Region"].unique())
    channels = st.sidebar.multiselect(
        "Select Channels (Unified)",
        options=df["Channel"].unique(),
        default=list(df["Channel"].unique())
    )

    # -----------------------------
    # Filter & Pivot
    # -----------------------------
    filt_df  = df[(df["Product"]==product)&(df["Region"]==region)&(df["Channel"].isin(channels))]
    pivot_df = filt_df.pivot(index="Date", columns="Channel",
                             values="Forecasted_Demand").fillna(0)

    # -----------------------------
    # Forecast Line Chart
    # -----------------------------
    st.subheader(f"Forecasted Demand for {product} in {region}")
    st.line_chart(pivot_df)

    # -----------------------------
    # Hyper‑local Demand Heatmap
    # -----------------------------
    st.subheader("Hyperlocal Demand Heatmap")
    heatmap_data = pd.DataFrame(
        np.random.randint(30,180,size=(5,5)),
        columns=[f"Zone {i}" for i in range(1,6)],
        index=[f"Store {i}" for i in range(1,6)]
    )
    st.dataframe(heatmap_data,use_container_width=True)

    # -----------------------------
    # Inventory Rebalancing Suggestion
    # -----------------------------
    st.subheader("AI‑Driven Inventory Rebalancing Suggestion")
    if heatmap_data.values.max() > 150:
        max_d   = heatmap_data.values.max()
        avg_d   = heatmap_data.values.mean()
        sugg    = max(20,int(np.ceil((max_d-avg_d)/10.0)*10))
        idx     = np.argmax(heatmap_data.values)
        zone_no = idx // heatmap_data.shape[1] + 1
        store_no= idx %  heatmap_data.shape[1] + 1
        st.markdown(
            f"<div style='background:#111;color:#fff;padding:16px;border-radius:8px;'>"
            f"<b>High demand detected in {region} – Zone {zone_no}.</b><br>"
            f"Suggest moving <b>{sugg} units</b> of <b>{product}</b> "
            f"from Warehouse {region} to Store {store_no}.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background:#d4edda;padding:16px;border-radius:8px;'>"
            f"Inventory levels are balanced for <b>{product}</b> in {region}.</div>",
            unsafe_allow_html=True
        )

    # -----------------------------
    # Forecast Accuracy Metrics
    # -----------------------------
    st.subheader("Forecast Accuracy Metrics")
    def simulate_truth_and_pred(df_):
        y_true = df_["Forecasted_Demand"].values + np.random.normal(0,8,len(df_))
        y_pred_un  = df_["Forecasted_Demand"].values
        y_pred_frag= y_pred_un + np.random.normal(0,18,len(df_))
        return y_true,y_pred_un,y_pred_frag

    def mape(y_true,y_pred):
        y_true,y_pred = np.array(y_true),np.array(y_pred)
        mask = y_true!=0
        return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

    yt,ypu,ypf = simulate_truth_and_pred(filt_df)
    mape_un    = mape(yt,ypu)
    mape_frag  = mape(yt,ypf)

    c1,c2 = st.columns(2)
    c1.metric("MAPE (Unified Model)",     f"{mape_un:.1f}%")
    c2.metric("MAPE (Fragmented Systems)",f"{mape_frag:.1f}%")

    # -----------------------------
    # Business Impact Summary
    # -----------------------------
    st.subheader("Business Impact Summary")
    stockout   = max(0, round((mape_frag-mape_un)*1.5,1))
    logistics  = max(0, round((mape_frag-mape_un)*1.2,1))
    sales      = max(0, round((mape_frag-mape_un)*1.0,1))
    st.markdown(f"""
    <ul style='font-size:16px;'>
      <li><b>{stockout}%</b> reduction in stockouts (Unified vs Fragmented)</li>
      <li><b>{logistics}%</b> improvement in logistics cost</li>
      <li><b>{sales}%</b> uplift in multi‑channel sales</li>
      <li>Real‑time integration across all channels and 3rd‑party data</li>
    </ul>
    """, unsafe_allow_html=True)
