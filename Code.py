# ===================================================
# IMPORTS
# ===================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import spearmanr

# Custom helper functions to read WCS experiment data
from wcs_helper_functions import readFociData, readClabData, readChipData

# ===================================================
# CONFIGURATION PARAMETERS
# ===================================================
NUM_CLUSTERS = 11                # Number of clusters for KMeans/GMM
MIN_SPEAKER_CONSENSUS = 10       # Minimum number of speakers required for a term
PERMUTATION_ITERS = 5000         # Number of iterations for permutation test
RANDOM_STATE = 42                # Random seed for reproducibility

# Names of the 11 anchor colors
ANCHOR_NAMES = ['BLACK','WHITE','RED','GREEN','YELLOW','BLUE','BROWN','PURPLE','PINK','ORANGE','GREY']

# ---------------------------------------------------
# English WCS anchors (with LAB coordinates and evolutionary stage)
English_ANCHORS = {
    'BLACK':  {'coord': [5.00,0.00,0.00],  'Stage': 1},
    'WHITE':  {'coord': [95.00,0.00,0.00], 'Stage': 1},
    'RED':    {'coord': [39.7,44.84,10.02], 'Stage': 2},
    'GREEN':  {'coord': [53.5,-32.64,18.67], 'Stage': 3},
    'YELLOW': {'coord': [75.6,6.45,40.89],   'Stage': 3},
    'BLUE':   {'coord': [54.2,-10.26,-23.46], 'Stage': 4},
    'BROWN':  {'coord': [44.1,14.57,17.28],   'Stage': 5},
    'PURPLE': {'coord': [45.3,17.1,-19.72],   'Stage': 6},
    'PINK':   {'coord': [60.4,28.17,3.63],    'Stage': 6},
    'ORANGE': {'coord': [59.1,37.39,33.94],   'Stage': 6},
    'GREY':   {'coord': [54.0,-0.88,1.22],    'Stage': 6},
}

# ---------------------------------------------------
# Munsell anchors (alternative color system)
MUNSELL_ANCHORS = {
    'BLACK': {'coord':[5,0,0], 'Stage':1},
    'WHITE': {'coord':[95,0,0], 'Stage':1},
    'RED': {'coord':[45,68,40], 'Stage':2},
    'GREEN': {'coord':[50,-60,40], 'Stage':3},
    'YELLOW': {'coord':[85,-5,85], 'Stage':3},
    'BLUE': {'coord':[38,12,-48], 'Stage':4},
    'BROWN': {'coord':[35,18,30], 'Stage':5},
    'PURPLE': {'coord':[40,45,-30], 'Stage':6},
    'PINK': {'coord':[75,35,10], 'Stage':6},
    'ORANGE': {'coord':[65,45,55], 'Stage':6},
    'GREY': {'coord':[50,0,0], 'Stage':6}
}

# Predefined colors for plotting bars
ANCHOR_COLORS = {
    'BLACK':'black',
    'WHITE':'lightgrey',
    'RED':'red',
    'GREEN':'green',
    'YELLOW':'yellow',
    'BLUE':'blue',
    'BROWN':'saddlebrown',
    'PURPLE':'purple',
    'PINK':'deeppink',
    'ORANGE':'orange',
    'GREY':'grey'
}

# ===================================================
# DISTANCE FUNCTION
# ===================================================
def delta_e_lab(a, b):
    """
    Compute Euclidean distance (Delta E) between two LAB points.
    
    Parameters:
    a, b : list or array
        LAB coordinates
    
    Returns:
    float : Euclidean distance
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.linalg.norm(a - b))

# ===================================================
# DATA LOADING
# ===================================================
def load_and_prep_data(path="./WCS_data_core"):
    """
    Load WCS experiment data and prepare a cleaned term database.

    Returns:
    - all_pts: list of all LAB points
    - term_db: list of dicts, each containing term info and coordinates
    """
    # Load foci (speaker term coordinates), clab (chip→LAB mapping), and chip info
    foci = readFociData(os.path.join(path,"foci-exp.txt"))
    clab = readClabData(os.path.join(path,"cnum-vhcm-lab-new.txt"))
    chip = readChipData(os.path.join(path,"chip.txt"))[0]

    all_pts = []
    term_db = []

    # Iterate over languages and speakers
    for lang, speakers in foci.items():
        term_map = {}
        for spk, terms in speakers.items():
            for t, fociList in terms.items():
                term_map.setdefault(t, [])
                for f in fociList:
                    code = f.replace(":","")
                    if code in chip and chip[code] in clab:
                        L,a,b = clab[chip[code]]
                        try:
                            pt = [float(L),float(a),float(b)]
                        except:
                            continue
                        term_map[t].append(pt)
                        all_pts.append(pt)

        # Filter terms by minimum speaker consensus
        for term, coords in term_map.items():
            if len(coords) >= MIN_SPEAKER_CONSENSUS:
                term_db.append({
                    "Lang_ID": lang,
                    "Term_Label": term,
                    "Coords": np.array(coords)
                })

    return np.array(all_pts), term_db

# ===================================================
# CLUSTER → ANCHOR MAPPING
# ===================================================
def label_clusters_by_anchors(centers, anchor_dict):
    """
    Map each cluster center to the nearest anchor.
    
    Parameters:
    - centers: array of cluster centers
    - anchor_dict: dictionary of anchor coordinates
    
    Returns:
    dict mapping cluster IDs to closest anchor info
    """
    out = {}
    for cid, c in enumerate(centers):
        best, bestd = None, 1e9
        for name, info in anchor_dict.items():
            d = delta_e_lab(c, info["coord"])
            if d < bestd:
                bestd = d
                best = (name, info["Stage"])
        out[cid] = {
            "Name": best[0],
            "Stage": best[1],
            "Anchor_Dist": float(bestd),
            "Center": c.tolist()
        }
    return out

# ===================================================
# DISPERSION COMPUTATION
# ===================================================
def compute_term_dispersion(term_db, model, cluster_map):
    """
    Compute mean dispersion (ΔE) of each term from its centroid.
    
    Returns:
    DataFrame with columns: Lang_ID, Native_Term, Universal_Category, Stage, Variability, nTokens
    """
    rows = []
    for e in term_db:
        pts = e["Coords"]
        if pts.size == 0:
            continue
        centroid = pts.mean(axis=0)                 # compute centroid of term
        cl = model.predict(centroid.reshape(1,-1))[0]  # cluster assignment
        cat = cluster_map[cl]["Name"]
        stage = cluster_map[cl]["Stage"]
        dists = [delta_e_lab(p, centroid) for p in pts]  # distances to centroid
        rows.append({
            "Lang_ID": e["Lang_ID"],
            "Native_Term": e["Term_Label"],
            "Universal_Category": cat,
            "Stage": stage,
            "Variability": float(np.mean(dists)),
            "nTokens": len(pts)
        })
    return pd.DataFrame(rows)

# ===================================================
# PERMUTATION TEST
# ===================================================
def permutation_test(df, stage_map, iters=5000, seed=42):
    """
    Perform a permutation test for correlation between stage and dispersion.
    
    Returns:
    - observed Spearman correlation
    - p-value
    - null distribution
    """
    df2 = df.copy()
    df2["Stage"] = df2["Universal_Category"].map(stage_map)
    stages = df2["Stage"].values.astype(float)
    vals = df2["Variability"].values.astype(float)
    
    # Observed correlation
    rho_obs, _ = spearmanr(stages, vals)
    
    # Null distribution
    rng = np.random.default_rng(seed)
    null = np.zeros(iters)
    for i in range(iters):
        perm = rng.permutation(stages)
        null[i], _ = spearmanr(perm, vals)
    
    # Compute permutation p-value
    p = np.mean(np.abs(null) >= np.abs(rho_obs))
    return float(rho_obs), float(p), null

# ===================================================
# PLOTTING
# ===================================================
def make_plots(df, anchors, method_name, anchor_name, prefix):
    """
    Generate violin and bar plots of dispersion.
    
    Returns: list of saved file paths
    """
    sns.set(style="whitegrid")
    stage_map = {k: anchors[k]["Stage"] for k in anchors}
    df = df.copy()
    df["Stage"] = df["Universal_Category"].map(stage_map)

    os.makedirs("results", exist_ok=True)
    plot_files = []

    # --- Violin plot by evolutionary stage ---
    plt.figure(figsize=(12,6))
    sns.violinplot(data=df, x="Stage", y="Variability", inner="box", cut=0)
    plt.title(f"Dispersion by evolutionary stage ({method_name.upper()}, {anchor_name} anchors)")
    plt.xlabel("Evolutionary Stage")
    plt.ylabel("Mean Dispersion (ΔE)")
    plt.tight_layout()
    f1 = f"{prefix}_stage_violin.png"
    plt.savefig(f1, dpi=200)
    plt.close()
    plot_files.append(f1)

    # --- Category mean barplot ---
    cat_means = df.groupby("Universal_Category")["Variability"].mean().reset_index()
    cat_means["Universal_Category"] = pd.Categorical(cat_means["Universal_Category"],
                                                     categories=ANCHOR_NAMES, ordered=True)
    cat_means = cat_means.sort_values("Universal_Category")
    colors = [ANCHOR_COLORS[cat] for cat in cat_means["Universal_Category"]]

    plt.figure(figsize=(12,6))
    sns.barplot(data=cat_means, x="Universal_Category", y="Variability", palette=colors)
    plt.title(f"Category mean dispersion ({method_name.upper()}, {anchor_name} anchors)")
    plt.xlabel("Universal Category")
    plt.ylabel("Mean Dispersion (ΔE)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    f2 = f"{prefix}_cat_means_color.png"
    plt.savefig(f2, dpi=200)
    plt.close()
    plot_files.append(f2)

    return plot_files

# ===================================================
# HYPOTHESIS TESTING
# ===================================================
def test_late_vs_early(df):
    """
    Compare mean dispersion of early vs late stage terms.
    
    Returns a dictionary summarizing the hypothesis result.
    """
    stage_means = df.groupby("Stage")["Variability"].mean()
    early_mean = stage_means.loc[stage_means.index<=2].mean()
    late_mean  = stage_means.loc[stage_means.index>=5].mean()
    hypothesis = "Late terms more dispersed" if late_mean > early_mean else "No support"
    return {
        "Early_Mean": early_mean,
        "Late_Mean": late_mean,
        "Hypothesis_Result": hypothesis
    }

# ===================================================
# MAIN ANALYSIS FUNCTION
# ===================================================
def run_analysis(X, term_db, anchors, method="kmeans"):
    """
    Run full analysis pipeline:
    - Cluster terms (KMeans or GMM)
    - Map clusters to anchors
    - Compute dispersion
    - Perform permutation test
    - Generate plots
    - Test late vs early hypothesis
    - Save results
    """
    anchor_name = "English" if anchors is English_ANCHORS else "Munsell"
    print(f"\n=== Running {method.upper()} ({anchor_name} anchors) ===")

    # --- Clustering ---
    if method=="kmeans":
        model = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        model.fit(X)
        centers = model.cluster_centers_
    else:
        model = GaussianMixture(n_components=NUM_CLUSTERS, random_state=RANDOM_STATE)
        model.fit(X)
        centers = model.means_

    # --- Map clusters to nearest anchor ---
    cmap = label_clusters_by_anchors(centers, anchors)

    # --- Compute term dispersions ---
    df = compute_term_dispersion(term_db, model, cmap)

    # --- Permutation correlation test ---
    stage_map = {name: anchors[name]["Stage"] for name in anchors}
    rho, p, null = permutation_test(df, stage_map, iters=PERMUTATION_ITERS)

    # --- Plotting ---
    prefix = f"results/{method}_{anchor_name.lower()}"
    plot_files = make_plots(df, anchors, method, anchor_name, prefix)

    # --- Test hypothesis: late vs early terms ---
    hypo = test_late_vs_early(df)
    print(f"Hypothesis test: {hypo}")

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"{prefix}_term_dispersion.csv", index=False)
    with open(f"{prefix}_cluster_map.json","w") as f: 
        json.dump(cmap,f,indent=2)
    json.dump({"rho":rho, "p_perm":p, **hypo}, open(f"{prefix}_summary.json","w"), indent=2)

    summary = {"rho":rho, "p_perm":p, **hypo}
    return df, cmap, summary, plot_files

# ===================================================
# MAIN ENTRY POINT
# ===================================================
if __name__=="__main__":
    print("Loading WCS data...")
    X, term_db = load_and_prep_data()
    print(f"Loaded {len(X)} points, {len(term_db)} terms.")

    results = []
    all_plots = []

    # Run analysis for both anchor sets and both methods
    for anchors in [English_ANCHORS, MUNSELL_ANCHORS]:
        for method in ["kmeans","gmm"]:
            df, cmap, summary, plots = run_analysis(X, term_db, anchors, method)
            summary["Anchor_Set"] = "English" if anchors is English_ANCHORS else "Munsell"
            summary["Method"] = method
            results.append(summary)
            all_plots.extend(plots)

    # Save final summary table
    final_table = pd.DataFrame(results)
    final_table = final_table[["Anchor_Set","Method","Early_Mean","Late_Mean","Hypothesis_Result","p_perm"]]
    final_table.to_csv("results/final_hypothesis_summary.csv", index=False)
    print("\nAll analyses complete. Final summary table saved as 'results/final_hypothesis_summary.csv'.")

    # Print paths to generated plots
    print("\nGenerated plots saved at:")
    for p in all_plots:
        print(p)
