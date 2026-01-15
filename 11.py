import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
import streamlit.components.v1 as components  # 替换 st_folium
from folium.plugins import HeatMap
from datetime import datetime, timedelta, time
from sklearn.linear_model import LinearRegression  # 用于简单预测

# --- 1. 页面配置 ---
st.set_page_config(page_title="蟹工机械 - 舟山综合版 v1.0", layout="wide", initial_sidebar_state="expanded")


# --- 2. 核心数据初始化 ---
def init_data():
    if 'crab_data' not in st.session_state:
        data = []
        fishery_spots = {
            "沈家门渔场": (29.9430, 122.3020),
            "朱家尖海域": (29.8950, 122.3880),
            "岱山渔场": (30.2520, 122.1550),
            "桃花岛海域": (29.7550, 122.2010)
        }
        # 模拟 2026年1月 的数据
        base_date = datetime(2026, 1, 15)
        for i in range(20, -1, -1):
            curr_date = (base_date - timedelta(days=i)).date()
            # 模拟有些天数没有捕捞（无数据天数）
            if i in [5, 12, 18]: continue

            count = 15
            for j in range(count):
                spot_name = np.random.choice(list(fishery_spots.keys()))
                base_lat, base_lon = fishery_spots[spot_name]
                weight = round(float(np.random.normal(0.65, 0.1)), 2)
                weight_g = weight * 500
                volume = round(float(weight_g / np.random.uniform(0.7, 0.95)), 1)
                density = round(float(weight_g / volume), 3)

                # 品质分级
                if density > 0.85:
                    q = "💎 极品"
                elif density > 0.70:
                    q = "✅ 优良"
                else:
                    q = "⚠️ 偏瘦"

                data.append({
                    'ID': f"ZS{curr_date.strftime('%m%d')}_{j + 1}",
                    'Date': curr_date,
                    'Hour': np.random.randint(5, 18),
                    'Fishery': spot_name,
                    'Gender': np.random.choice(['公', '母']),
                    'Weight_Jin': weight,
                    'Volume_cm3': volume,
                    'Density': density,
                    'Quality': q,
                    'Latitude': float(base_lat + np.random.uniform(-0.03, 0.03)),
                    'Longitude': float(base_lon + np.random.uniform(-0.03, 0.03)),
                    'Is_Dead': np.random.choice(["是", "否"], p=[0.05, 0.95]),
                    'Missing_Leg': np.random.choice(["是", "否"], p=[0.1, 0.9])
                })
        st.session_state.crab_data = pd.DataFrame(data)


init_data()
df = st.session_state.crab_data

# --- 3. 侧边栏 ---
with st.sidebar:
    st.title("🦀 蟹工机械 v1.0")
    st.markdown("---")
    page = st.radio("功能模块", ["📊 实时看板", "🧪 质量深度分析", "⚓ 沿海捕捞地图", "🔮 预测与分析", "📝 数据库管理"],
                    label_visibility="collapsed")
    selected_date = st.date_input("全局日期筛选", value=df['Date'].max())
    st.markdown("---")
    st.caption("Powered by Streamlit & Plotly")

# --- 页面 1: 实时看板 ---
if page == "📊 实时看板":
    df_today = df[df['Date'] == selected_date].sort_values("Hour")
    st.title(f"🌊 {selected_date} 捕捞分析")

    if df_today.empty:
        st.warning("⚠️ 该日期无捕捞记录，请选择其他日期。")
    else:
        # KPI 栏 - 增加颜色和图标美观
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("今日捕捞量", f"{len(df_today)} 只", delta=None, label_visibility="visible")
        m2.metric("平均密度", f"{df_today['Density'].mean():.3f}",
                  delta=f"{df_today['Density'].mean() - df['Density'].mean():.3f} vs 总体", delta_color="normal")
        m3.metric("极品率", f"{(df_today['Density'] > 0.85).mean():.1%}", delta=None)
        m4.metric("最活跃渔场", df_today['Fishery'].mode()[0], delta=None)

        st.divider()

        c1, c2 = st.columns([3, 2])
        with c1:
            # 1. 密度时间折线图 - 增加平滑和阴影
            hourly_data = df_today.groupby("Hour")["Density"].mean().reset_index()
            fig_line = px.line(hourly_data, x="Hour", y="Density", title="今日密度变化曲线",
                               markers=True, color_discrete_sequence=['#4682B4'])
            fig_line.update_traces(line=dict(shape='spline', smoothing=1.3))  # 平滑曲线
            fig_line.add_trace(go.Scatter(x=hourly_data['Hour'], y=hourly_data['Density'],
                                          fill='tozeroy', fillcolor='rgba(70,130,180,0.2)',
                                          line=dict(color='rgba(255,255,255,0)')))  # 阴影填充
            st.plotly_chart(fig_line, use_container_width=True)

            # 2. 回归的饼状图 - 增加拉出和标签
            fig_pie = px.pie(df_today, names='Quality', title="今日品质占比分析",
                             color='Quality', color_discrete_map={
                    "💎 极品": "#FFD700", "✅ 优良": "#90EE90", "⚠️ 偏瘦": "#FF6347"
                }, hole=0.3)  # 增加甜甜圈效果
            fig_pie.update_traces(pull=[0.1 if q == "💎 极品" else 0 for q in df_today['Quality'].unique()])  # 拉出极品部分
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("📋 详细记录表")
            event = st.dataframe(
                df_today[['ID', 'Fishery', 'Density', 'Weight_Jin', 'Quality']],
                use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row"
            )

            if event.selection.rows:
                row = df_today.iloc[event.selection.rows[0]]
                with st.expander(f"🆔 档案: {row['ID']} - 单独分析", expanded=True):  # 使用expander增加实用性和美观
                    st.write(f"**来自渔场:** {row['Fishery']} | **性别:** {row['Gender']}")
                    st.write(f"**重量指标:** {row['Weight_Jin']}斤 / {row['Volume_cm3']}cm³")
                    st.write(f"**健康状态:** 断腿({row['Missing_Leg']}) / 死亡({row['Is_Dead']})")
                    st.info(f"**体密度判定: {row['Quality']} ({row['Density']})**")

                    # 新增: 对螃蟹的单独分析 - 实用性增强
                    avg_density = df_today['Density'].mean()
                    delta_density = row['Density'] - avg_density
                    color = "green" if delta_density > 0 else "red"
                    st.metric("密度 vs 今日平均", f"{row['Density']:.3f}", delta=f"{delta_density:.3f}",
                              delta_color="normal" if delta_density > 0 else "inverse")

                    # 小型进度条显示密度水平
                    st.progress(row['Density'], text="密度水平 (0-1)")

                    # 简单比较图 - 与今日平均和极值比较
                    compare_data = pd.DataFrame({
                        '指标': ['密度', '重量 (斤)', '体积 (cm³)'],
                        '本只': [row['Density'], row['Weight_Jin'], row['Volume_cm3']],
                        '今日平均': [df_today['Density'].mean(), df_today['Weight_Jin'].mean(),
                                     df_today['Volume_cm3'].mean()],
                        '今日最高': [df_today['Density'].max(), df_today['Weight_Jin'].max(),
                                     df_today['Volume_cm3'].max()],
                        '今日最低': [df_today['Density'].min(), df_today['Weight_Jin'].min(),
                                     df_today['Volume_cm3'].min()]
                    })
                    fig_bar = px.bar(compare_data.melt(id_vars='指标'), x='variable', y='value', color='variable',
                                     title="本只 vs 今日统计", text='value', facet_col='指标', facet_col_wrap=1)
                    fig_bar.update_traces(textposition='outside')
                    fig_bar.update_layout(height=600)  # 增加高度以适应多个facet
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # 增强可视化: 添加雷达图展示多维度属性
                    radar_data = pd.DataFrame(dict(
                        r=[row['Density'], row['Weight_Jin'], 1 if row['Missing_Leg'] == "否" else 0,
                           1 if row['Is_Dead'] == "否" else 0],
                        theta=['密度', '重量', '完整腿', '存活']
                    ))
                    fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True, title="个体属性雷达图")
                    fig_radar.update_traces(fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line_color='red')
                    st.plotly_chart(fig_radar, use_container_width=True)

                    # 增强可视化: 添加仪表盘式密度显示
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=row['Density'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "密度仪表"},
                        delta={'reference': avg_density},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.7], 'color': "lightgray"},
                                {'range': [0.7, 0.85], 'color': "gray"},
                                {'range': [0.85, 1], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': row['Density']
                            }
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)

# --- 页面 2: 质量深度分析 ---
elif page == "🧪 质量深度分析":
    st.title("🧪 质量深度分析面板")
    dr = st.date_input("分析区间", value=(df['Date'].min(), df['Date'].max()))

    if len(dr) == 2:
        df_f = df[(df['Date'] >= dr[0]) & (df['Date'] <= dr[1])]

        # 修复：空数据天数拦截，不显示点击项
        if df_f.empty:
            st.error("🚫 所选区间内暂无数据，请重新选择日期。")
        else:
            c_main, c_top = st.columns([3, 1])
            with c_main:
                fig_s = px.scatter(df_f, x="Weight_Jin", y="Density", color="Quality",
                                   size="Volume_cm3", hover_name="ID", facet_col="Gender",
                                   title="重量-密度交叉分析 (按性别分面)")
                st.plotly_chart(fig_s, use_container_width=True)

            with c_top:
                st.subheader("🏆 密度排名")
                top_3 = df_f.sort_values("Density", ascending=False).head(3)
                for i, r in top_3.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**Top {top_3.index.get_loc(i) + 1}**")
                        st.write(f"{r['ID']} ({r['Density']})")
                        if st.button("查看体检表", key=f"top_{r['ID']}"):
                            st.session_state.deep_id = r['ID']

            if "deep_id" in st.session_state:
                target = df[df['ID'] == st.session_state.deep_id].iloc[0]
                st.success(f"📑 选定个体全量数据：{target['ID']}")
                st.table(pd.DataFrame([target]))

# --- 页面 3: 沿海地图 ---
elif page == "⚓ 沿海捕捞地图":
    st.title("⚓ 舟山捕捞点位分布图")
    df_m = df[df['Date'] == selected_date]
    if df_m.empty:
        st.warning("该日期无坐标数据")
    else:
        m = folium.Map(location=[29.98, 122.25], zoom_start=10)
        HeatMap([[r.Latitude, r.Longitude] for r in df_m.itertuples()]).add_to(m)
        for r in df_m.itertuples():
            folium.CircleMarker(
                [r.Latitude, r.Longitude], radius=5,
                popup=f"{r.ID}: {r.Quality}",
                color='red' if r.Is_Dead == "是" else 'blue', fill=True
            ).add_to(m)

        # 替换 st_folium 以兼容更多环境
        m.save("temp_map.html")
        with open("temp_map.html", "r", encoding= 'utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600, scrolling=True)

# --- 新页面 4: 预测与分析 (数字孪生风格) ---
# --- 新页面 4: 预测与分析 (数字孪生风格) ---
elif page == "🔮 预测与分析":
    st.title("🔮 捕捞预测与产量分析 (数字孪生)")
    st.markdown("此页面提供基于历史数据的综合分析和预测，模拟数字孪生系统，包括产量趋势、渔场分布、品质分解及未来预测。")

    # 准备数据
    df['Total_Weight'] = df['Weight_Jin']  # 假设产量以总重量斤为单位
    daily_yield = df.groupby('Date').agg({'Total_Weight': 'sum', 'ID': 'count'}).rename(columns={'ID': 'Count'})
    fishery_yield = df.groupby(['Fishery', 'Date'])['Total_Weight'].sum().unstack().fillna(0)
    quality_yield = df.groupby(['Quality', 'Date'])['Total_Weight'].sum().unstack().fillna(0)
    gender_yield = df.groupby(['Gender', 'Date'])['Total_Weight'].sum().unstack().fillna(0)

    # 1. 历史产量趋势图
    st.subheader("📈 历史产量趋势")
    fig_trend = px.line(daily_yield, x=daily_yield.index, y='Total_Weight', title="每日总产量 (斤) 趋势",
                        markers=True, line_shape='spline')
    fig_trend.add_bar(x=daily_yield.index, y=daily_yield['Count'], name='捕捞数量 (只)')
    st.plotly_chart(fig_trend, use_container_width=True)

    # 2. 渔场产量比较 (堆叠柱状图)
    st.subheader("🏞️ 渔场产量分布")
    fig_fishery = px.bar(fishery_yield.T, title="各渔场每日产量比较 (斤)", barmode='stack')
    st.plotly_chart(fig_fishery, use_container_width=True)

    # 3. 产量成分分析 (品质分解)
    st.subheader("🍎 产量成分分析 - 按品质")
    fig_quality = px.area(quality_yield.T, title="每日产量按品质分解 (斤)")
    st.plotly_chart(fig_quality, use_container_width=True)

    # 4. 产量成分分析 (性别分解)
    st.subheader("♂️♀️ 产量成分分析 - 按性别")
    fig_gender = px.bar(gender_yield.T, title="每日产量按性别分解 (斤)", barmode='group')
    st.plotly_chart(fig_gender, use_container_width=True)

    # 5. 相关性热力图 (变量间相关)
    st.subheader("🔗 变量相关性分析")
    corr_matrix = df[['Weight_Jin', 'Volume_cm3', 'Density', 'Hour']].corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=True, title="关键指标相关性热力图",
                            color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # 6. 预测明天捕捞 (简单线性回归模型)
    st.subheader("🔮 明天捕捞预测")
    # 准备预测数据
    dates = pd.to_datetime(daily_yield.index)
    X = np.array((dates - dates.min()).days).reshape(-1, 1)
    y = daily_yield['Total_Weight'].values
    model = LinearRegression().fit(X, y)
    next_day = len(X)  # 下一天
    pred_yield = model.predict([[next_day]])[0]
    pred_count = round(pred_yield / df['Weight_Jin'].mean())  # 估算数量

    # 渔场预测: 基于平均比例
    fishery_avg = df.groupby('Fishery')['Total_Weight'].sum() / df['Total_Weight'].sum()
    pred_fishery = fishery_avg * pred_yield

    # 新增: 模拟环境因素调整预测 (假设天气、潮汐等影响)
    # 假设天气影响: 台风或风暴减少产量20%; 潮汐上涌增加产量10%; 其他因素如红潮减少15%
    # 这里用随机模拟实际应集成真实API数据
    weather_factor = np.random.choice([0.8, 1.0, 1.1])  # 0.8:不良天气, 1.0:正常, 1.1:有利
    tide_factor = np.random.choice([0.9, 1.0, 1.2])  # 0.9:弱潮, 1.0:正常, 1.2:强上涌
    env_factor = np.random.choice([0.85, 1.0])  # 0.85:红潮等负面, 1.0:正常
    adjusted_pred_yield = pred_yield * weather_factor * tide_factor * env_factor
    adjusted_pred_count = round(adjusted_pred_yield / df['Weight_Jin'].mean())

    st.metric("基础预测总产量 (斤)", f"{pred_yield:.2f}", delta=None)
    st.metric("基础预测捕捞数量 (只)", f"{pred_count}", delta=None)
    st.metric("考虑环境调整后总产量 (斤)", f"{adjusted_pred_yield:.2f}",
              delta=f"{adjusted_pred_yield - pred_yield:.2f}")
    st.metric("考虑环境调整后捕捞数量 (只)", f"{adjusted_pred_count}", delta=f"{adjusted_pred_count - pred_count}")

    # 预测图
    fig_pred = px.line(daily_yield, x=daily_yield.index, y='Total_Weight', title="产量趋势与预测")
    future_date = dates.max() + timedelta(days=1)
    fig_pred.add_scatter(x=[future_date], y=[pred_yield], mode='markers+text', text=['基础预测'], name='基础预测')
    fig_pred.add_scatter(x=[future_date], y=[adjusted_pred_yield], mode='markers+text', text=['调整预测'],
                         name='调整预测')
    st.plotly_chart(fig_pred, use_container_width=True)

    # 预测渔场分布饼图
    fig_pred_pie = px.pie(values=pred_fishery.values, names=pred_fishery.index, title="预测明天各渔场产量占比")
    st.plotly_chart(fig_pred_pie, use_container_width=True)

    # 7. 额外分析: 健康状态分布
    st.subheader("🩺 健康状态分析")
    health_data = df.groupby(['Is_Dead', 'Missing_Leg'])['ID'].count().reset_index(name='Count')
    fig_health = px.bar(health_data, x='Is_Dead', y='Count', color='Missing_Leg', title="死亡与断腿分布")
    st.plotly_chart(fig_health, use_container_width=True)

    # 8. 时段产量分析 (数字孪生 - 模拟最佳捕捞时间)
    st.subheader("⏰ 时段产量分析与最佳建议")
    hourly_yield = df.groupby('Hour')['Total_Weight'].mean()
    fig_hourly = px.bar(hourly_yield, title="平均每小时产量 (斤)")
    best_hour = hourly_yield.idxmax()
    fig_hourly.add_annotation(x=best_hour, y=hourly_yield.max(), text="最佳时段", showarrow=True)
    st.plotly_chart(fig_hourly, use_container_width=True)

    # 9. 密度分布直方图
    st.subheader("📊 密度分布统计")
    fig_hist = px.histogram(df, x='Density', color='Quality', title="整体密度分布 (数字孪生模拟)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 10. 3D 散点图 (纬度、经度、密度 - 模拟空间分布)
    st.subheader("🌐 空间密度分布 (3D 数字孪生)")
    fig_3d = px.scatter_3d(df, x='Longitude', y='Latitude', z='Density', color='Quality',
                           size='Weight_Jin', title="捕捞点位3D密度图")
    st.plotly_chart(fig_3d, use_container_width=True)

    # 新增: 环境因素影响分析
    st.subheader("🌤️ 环境因素对捕捞的影响分析")
    st.markdown("""
    基于舟山渔场的历史和科学研究，以下是主要环境因素对蟹捕捞的影响分析（模拟数字孪生考虑现实变量）：

    - **天气因素（台风、风暴）**：
      - **影响因子**：强风、暴雨导致水体垂直混合，降低海面温度（SST），增加营养物输入。
      - **如何影响**：台风如“In-fa”可使SST下降2-5°C，短期内减少捕捞活动（安全风险），但后期叶绿素a（Chl-a）浓度增加，促进浮游植物生长，提升蟹食物链丰度。负面：极端天气中断捕捞，增加蟹死亡率。调整系数模拟：不良天气减少产量20%。

    - **潮汐因素（上涌、潮差）**：
      - **影响因子**：舟山附近季节性上涌（5月开始，7-8月峰值），带来冷水和营养物。
      - **如何影响**：上涌增强OPP（海洋初级生产力）和Chl-a，促进蟹栖息地营养丰富，提高捕捞产量。低潮时暴露浅滩，便于捕捞；高潮时水流强，影响网具。调整系数模拟：强上涌增加产量20%。

    - **其他环境因素（盐度、温度、红潮、气候变化）**：
      - **盐度（SSS）**：台风后因雨水和河流输入下降，影响蟹渗透压调节；理想范围28-32 PSU。
      - **温度**：SST 5-31°C季节变化，高温（如气候变暖）减少蟹种多样性，负面影响捕捞。
      - **红潮**：有害藻华（如赤潮）毒害蟹群，减少资源；是主要负面因素。
      - **气候变化**：整体变暖可能降低蟹捕捞量，影响区域渔业可持续性。
      - **如何影响**：这些因素通过改变栖息地、食物可用性和蟹行为影响产量。调整系数模拟：负面事件减少产量15%。

    **注意**：当前预测已整合简单环境调整（随机模拟真实数据）。实际应用可接入天气API或潮汐数据源进行动态修正。
    """)
# --- 页面 5: 编辑 ---
elif page == "📝 数据库管理" :
    st.title("📝 核心数据库")
    st.data_editor(df, use_container_width=True, hide_index=True)