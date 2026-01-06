"""
Simple Streamlit Web App for Real Estate Price Prediction
==========================================================

Chạy app:
    streamlit run app.py

Nếu chưa có streamlit:
    pip install streamlit
"""

import streamlit as st
import numpy as np
import joblib
import keras
import pandas as pd

# Page config
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        lr_model = joblib.load('models/linear_regression_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        ann_model = keras.models.load_model('models/ann_model.keras')
        scaler = joblib.load('models/scaler.pkl')
        return lr_model, rf_model, ann_model, scaler, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

# Title
st.title("🏠 Real Estate Price Prediction")
st.markdown("### Dự đoán giá nhà sử dụng Machine Learning & Deep Learning")
st.markdown("---")

# Load models
lr_model, rf_model, ann_model, scaler, models_loaded = load_models()

if models_loaded:
    st.success("✅ Models loaded successfully!")
    
    # Create two columns for main layout
    main_col, map_col = st.columns([2, 1])
    
    # Sidebar for input
    st.sidebar.header("🏡 Thông Tin Ngôi Nhà")
    st.sidebar.markdown("Nhập các thông tin bên dưới:")
    
    # Input fields
    med_inc = st.sidebar.slider(
        "💰 Thu nhập khu vực (median income)",
        min_value=0.5,
        max_value=15.0,
        value=3.5,
        step=0.1,
        help="Đơn vị: $10,000/năm\n\n" 
    )
    
    house_age = st.sidebar.slider(
        "🏚️ Tuổi nhà",
        min_value=1,
        max_value=52,
        value=15,
        step=1,
        help="Số năm kể từ khi xây"
    )
    
    ave_rooms = st.sidebar.slider(
        "🚪 Số phòng trung bình",
        min_value=1,
        max_value=15,
        value=6,
        step=1,
        help="Trung bình số phòng/hộ"
    )
    
    ave_bedrms = st.sidebar.slider(
        "🛏️ Số phòng ngủ trung bình",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Trung bình phòng ngủ/hộ"
    )
    
    population = st.sidebar.number_input(
        "👥 Dân số khu vực",
        min_value=3,
        max_value=35682,
        value=1200,
        step=100,
        help="Số người trong block"
    )
    
    ave_occup = st.sidebar.slider(
        "👨‍👩‍👧‍👦 Số người/hộ",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Số người sống chung/hộ"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input(
            "📍 Latitude",
            min_value=32.5,
            max_value=42.0,
            value=34.05,
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        longitude = st.number_input(
            "📍 Longitude",
            min_value=-124.5,
            max_value=-114.3,
            value=-118.25,
            step=0.01,
            format="%.2f"
        )
    
    # Ocean proximity selection
    ocean_proximity = st.sidebar.selectbox(
        "🌊 Vị trí so với biển",
        options=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
        index=0,
        help="Khoảng cách từ nhà đến đại dương"
    )
    
    # Live preview map in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Preview Vị Trí")
    
    # Create a simple preview map
    import plotly.express as px
    
    preview_df = pd.DataFrame({
        'lat': [latitude],
        'lon': [longitude],
        'Location': ['Selected Location']
    })
    
    fig_preview = px.scatter_mapbox(
        preview_df,
        lat='lat',
        lon='lon',
        hover_name='Location',
        zoom=8,
        height=200,
        size=[10]
    )
    fig_preview.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False
    )
    st.sidebar.plotly_chart(fig_preview, use_container_width=True)
    
    # Predict button
    if st.sidebar.button("🔮 DỰ ĐOÁN GIÁ", type="primary", use_container_width=True):
        
        # Prepare features with one-hot encoding for ocean_proximity
        # One-hot encode ocean_proximity (4 features)
        ocean_inland = 1 if ocean_proximity == 'INLAND' else 0
        ocean_island = 1 if ocean_proximity == 'ISLAND' else 0
        ocean_near_bay = 1 if ocean_proximity == 'NEAR BAY' else 0
        ocean_near_ocean = 1 if ocean_proximity == 'NEAR OCEAN' else 0
        
        features = np.array([[
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude,
            ocean_inland, ocean_island, ocean_near_bay, ocean_near_ocean
        ]])
        
        features_scaled = scaler.transform(features)
        
        # Predictions
        price_lr = lr_model.predict(features_scaled)[0]
        price_rf = rf_model.predict(features_scaled)[0]
        price_ann = ann_model.predict(features_scaled, verbose=0)[0][0]
        
        # Average prediction
        avg_price = (price_lr + price_rf + price_ann) / 3
        
        # Display results
        st.markdown("## 💰 Kết Quả Dự Đoán")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Linear Regression",
                value=f"${price_lr*100000:,.0f}",
                delta=f"{((price_lr - avg_price)/avg_price)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                label="🌳 Random Forest",
                value=f"${price_rf*100000:,.0f}",
                delta=f"{((price_rf - avg_price)/avg_price)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                label="🧠 Neural Network",
                value=f"${price_ann*100000:,.0f}",
                delta=f"{((price_ann - avg_price)/avg_price)*100:.1f}%"
            )
        
        with col4:
            st.metric(
                label="⭐ Trung Bình",
                value=f"${avg_price*100000:,.0f}",
                delta="Recommended"
            )
        
        # Best model
        models = {
            'Linear Regression': price_lr,
            'Random Forest': price_rf,
            'Neural Network': price_ann
        }
        best_model = max(models, key=models.get)
        
        st.success(f"🏆 **Giá cao nhất**: {best_model} - ${models[best_model]*100000:,.0f}")
        st.info(f"💡 **Đề xuất**: Dựa trên performance trong quá khứ, **Random Forest** thường chính xác nhất với R² = 0.81")
        
        # Detailed breakdown
        st.markdown("### 📋 Chi Tiết Thông Tin")
        
        input_df = pd.DataFrame({
            'Feature': [
                'Thu nhập khu vực', 'Tuổi nhà', 'Số phòng TB',
                'Số phòng ngủ TB', 'Dân số', 'Số người/hộ',
                'Vĩ độ', 'Kinh độ', 'Vị trí biển'
            ],
            'Giá trị': [
                f"${med_inc*10000:,.0f}/năm",
                f"{house_age} năm",
                f"{ave_rooms:.1f} phòng",
                f"{ave_bedrms:.1f} phòng",
                f"{population:,} người",
                f"{ave_occup:.1f} người",
                f"{latitude:.2f}°",
                f"{longitude:.2f}°",
                ocean_proximity
            ]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Comparison chart
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Linear Reg', 'Random Forest', 'ANN', 'Average'],
                    y=[price_lr*100000, price_rf*100000, price_ann*100000, avg_price*100000],
                    marker_color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c'],
                    text=[f"${price_lr*100000:,.0f}", f"${price_rf*100000:,.0f}", 
                          f"${price_ann*100000:,.0f}", f"${avg_price*100000:,.0f}"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="So Sánh Giá Dự Đoán",
                xaxis_title="Model",
                yaxis_title="Giá ($)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Location info with detailed map
        st.markdown("### 🗺️ Vị Trí Trên Bản Đồ California")
        
        # Create a more detailed map with Plotly
        import plotly.graph_objects as go
        
        map_df = pd.DataFrame({
            'lat': [latitude],
            'lon': [longitude],
            'text': [f'Predicted Price: ${avg_price*100000:,.0f}'],
            'income': [med_inc],
            'age': [house_age]
        })
        
        fig_map = go.Figure(go.Scattermapbox(
            lat=map_df['lat'],
            lon=map_df['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=20,
                color='red',
                opacity=0.8
            ),
            text=map_df['text'],
            hovertemplate='<b>Vị Trí Ngôi Nhà</b><br>' +
                         'Latitude: %{lat:.2f}<br>' +
                         'Longitude: %{lon:.2f}<br>' +
                         '%{text}<br>' +
                         f'Thu nhập: ${med_inc*10000:,.0f}/năm<br>' +
                         f'Tuổi nhà: {house_age} năm<br>' +
                         f'Vị trí biển: {ocean_proximity}' +
                         '<extra></extra>'
        ))
        
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=latitude, lon=longitude),
                zoom=9
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            height=500
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Add reference locations
        with st.expander("📍 Tham khảo các thành phố California"):
            ref_cities = pd.DataFrame({
                'Thành phố': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento', 'San Jose'],
                'Latitude': [34.05, 37.77, 32.72, 38.58, 37.34],
                'Longitude': [-118.24, -122.42, -117.16, -121.49, -121.89],
                'Giá TB': ['$650K', '$1.2M', '$750K', '$450K', '$1.0M']
            })
            st.dataframe(ref_cities, use_container_width=True, hide_index=True)
        
else:
    st.error("❌ Models chưa được train. Vui lòng chạy notebook trước!")
    st.info("💡 Chạy lệnh: `jupyter notebook real_estate_price_prediction.ipynb`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏠 <strong>Real Estate Price Prediction Project</strong></p>
    <p>Powered by Machine Learning & Deep Learning</p>
    <p>Models: Linear Regression | Random Forest | Neural Network</p>
</div>
""", unsafe_allow_html=True)
