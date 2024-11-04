from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # Generate initial dataset
    X = np.random.uniform(0, 1, N)
    Y = np.random.normal(mu, np.sqrt(sigma2), N)

    # Fit linear regression
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Create first plot - scatter with fitted line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Data Points')
    X_line = np.linspace(0, 1, 100)
    Y_line = intercept + slope * X_line
    plt.plot(X_line, Y_line, color='red', label='Fitted Line')
    plt.xlabel('x')
    plt.ylabel('Y')
    plt.title(f'Linear Fit: y = {intercept:.2f} + {slope:.2f}x')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Run simulations
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        Y_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        
        sim_model = LinearRegression()
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model.fit(X_sim_reshaped, Y_sim)
        
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Create histogram plot
    plt.figure(figsize=(10, 6))
    
    # Plot histograms with more bins and transparency
    plt.hist(slopes, bins=20, alpha=0.5, color='blue', label='Slopes')
    plt.hist(intercepts, bins=20, alpha=0.5, color='orange', label='Intercepts')
    
    # Add vertical lines for the calculated slope and intercept
    plt.axvline(slope, color='blue', linestyle='--', label=f'Slope: {slope:.2f}')
    plt.axvline(intercept, color='orange', linestyle='--', label=f'Intercept: {intercept:.2f}')
    
    plt.title('Histogram of Slopes and Intercepts')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Calculate proportions (converting to percentage and rounding to 2 decimal places)
    slope_more_extreme = round(sum(abs(s) > abs(slope) for s in slopes) / S * 100, 2)
    intercept_more_extreme = round(sum(abs(i) > abs(intercept) for i in intercepts) / S * 100, 2)

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)
        
        return render_template("index.html", 
                             plot1=plot1, 
                             plot2=plot2,
                             slope_extreme=slope_extreme, 
                             intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)