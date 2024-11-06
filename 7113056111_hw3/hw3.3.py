import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sklearn.svm import LinearSVC

# Step 1: Generate random points
st.title("3D SVM Classification with Adjustable Distance Threshold and Plane Tilt")
st.write("調整距離閾值來動態分類數據並即時顯示結果，並調整平面傾斜角度。")

# Set seed and parameters
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Calculate distances from origin
distances = np.sqrt(x1**2 + x2**2)

# Step 2: Distance threshold slider for classification
threshold = st.slider("分類距離閾值", min_value=1.0, max_value=15.0, value=4.0, step=0.5)

# Assign labels based on adjustable distance threshold
Y = np.where(distances < threshold, 0, 1)

# Define Gaussian function for x3
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# Calculate x3 as Gaussian function of x1 and x2
x3 = gaussian_function(x1, x2)

# Combine features into a single matrix X
X = np.column_stack((x1, x2, x3))

# Step 3: Train a LinearSVC to find the separating hyperplane
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# Step 4: Plane tilt slider to adjust z-axis offset of the hyperplane
tilt_angle = st.slider("調整平面傾斜角度", min_value=-30.0, max_value=30.0, value=0.0, step=1.0)

# Create 3D scatter plot with separating hyperplane
fig = go.Figure()

# Plot points for each class
for label, color in zip([0, 1], ['blue', 'red']):
    fig.add_trace(go.Scatter3d(
        x=X[Y == label, 0],
        y=X[Y == label, 1],
        z=X[Y == label, 2],
        mode='markers',
        marker=dict(size=4, color=color),
        name=f'Class {label}'
    ))

# Generate the hyperplane surface with tilt angle adjustment
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
# Tilt adjustment: modifying z by adding tilt_angle times normalized distance on xy-plane
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2] + tilt_angle * (xx + yy) / np.sqrt(xx**2 + yy**2)

# Add hyperplane surface to the figure
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='gray', opacity=0.5, showscale=False))

# Set plot titles and labels
fig.update_layout(
    title="3D 隨機數據分佈與 SVM 超平面",
    scene=dict(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="x3"
    )
)

# Display the 3D plot
st.plotly_chart(fig)
