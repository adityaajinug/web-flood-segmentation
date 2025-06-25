import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io

# Import your existing utility functions
from utils.segmentation_utils import evaluate_segmentation

def apply_kmeans(image, k):
    """Apply K-means clustering to segment the image"""
    # Reshape image to 2D array of pixels
    data = image.reshape(-1, 3)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    
    # Reshape back to image dimensions
    labels = labels.reshape(image.shape[:2])
    
    # Create segmented image
    segmented_img = np.zeros_like(image)
    for i in range(k):
        mask = (labels == i)
        segmented_img[mask] = kmeans.cluster_centers_[i]
    
    return segmented_img.astype(np.uint8), labels

def binarize_mask(mask, threshold=127):
    """Convert mask to binary (0 and 255)"""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    binary_mask = np.where(mask > threshold, 255, 0).astype(np.uint8)
    return binary_mask

def find_best_k(image, mask=None, k_range=(2, 10)):
    """Find the best K value using elbow method or evaluation metrics"""
    k_values = range(k_range[0], k_range[1] + 1)
    
    if mask is not None:
        # Use evaluation metrics to find best K
        binary_mask = binarize_mask(mask)
        best_k = 2
        best_score = 0
        
        scores = []
        for k in k_values:
            segmented_img, labels = apply_kmeans(image, k)
            resized_mask = cv2.resize(binary_mask, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_NEAREST)
            metrics, _ = evaluate_segmentation(labels, resized_mask, k)
            score = metrics['F1-Score']
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k, scores
    else:
        # Use elbow method with inertia
        inertias = []
        for k in k_values:
            data = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        best_k = k_values[0]
        if len(inertias) > 2:
            # Find the point with maximum decrease in inertia
            decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            best_k = k_values[np.argmax(decreases)]
        
        return best_k, inertias

def calculate_flood_area(labels, flood_label, pixel_size_m2=1.0):
    """Calculate flood area in square meters"""
    # For binary segmentation, assume smaller cluster is flood
    # unique_labels = np.unique(labels)
    # cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    
    # # Assume flood is the minority class (usually smaller area)
    # flood_label = unique_labels[np.argmin(cluster_sizes)]
    # flood_pixels = np.sum(labels == flood_label)
    # flood_area_m2 = flood_pixels * pixel_size_m2

    flood_pixels = np.sum(labels == flood_label)
    flood_area_m2 = flood_pixels * pixel_size_m2
    return flood_area_m2, flood_pixels, flood_label

def create_flood_overlay(original_image, labels, flood_label, alpha=0.3):
    """Create an overlay showing flood areas on the original image"""
    overlay = original_image.copy()
    flood_mask = (labels == flood_label)
    
    # Create colored overlay for flood areas (red)
    overlay[flood_mask] = [255, 0, 0]  # Red for flood
    
    # Blend with original image
    result = cv2.addWeighted(original_image, 1-alpha, overlay, alpha, 0)
    return result

def main():
    st.set_page_config(page_title="Flood Segmentation Tool", layout="wide")
    
    # Header
    st.title("🌊 UAV Flood Segmentation Tool")
    st.markdown("""
    **Upload UAV images for flood segmentation analysis**
    - Ground truth masks are optional for evaluation
    - Automatic K-value detection or manual selection
    - Flood area calculation and visualization
    """)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Parameters")
    
    # K-means parameters
    use_auto_k = st.sidebar.checkbox("🔍 Auto-find best K", value=True, 
                                   help="Automatically find the optimal number of clusters")
    
    if not use_auto_k:
        k_value = st.sidebar.slider("K value for clustering", min_value=2, max_value=15, value=3)
    else:
        st.sidebar.subheader("K Search Range")
        k_range_min = st.sidebar.slider("Min K", min_value=2, max_value=10, value=2)
        k_range_max = st.sidebar.slider("Max K", min_value=3, max_value=15, value=8)
    
    # Area calculation parameters
    st.sidebar.subheader("📏 Area Calculation")
    pixel_size_m2 = st.sidebar.number_input(
        "Pixel size (m²/pixel)", 
        min_value=0.001, 
        max_value=10.0, 
        value=1.0, 
        step=0.001,
        help="Ground sampling distance in square meters per pixel"
    )
    
    # Visualization parameters
    st.sidebar.subheader("🎨 Visualization")
    overlay_alpha = st.sidebar.slider("Overlay transparency", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    
    # File upload section
    st.header("📁 File Upload")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 UAV Image (Required)")
        uploaded_image = st.file_uploader(
            "Choose UAV image", 
            type=['png', 'jpg', 'jpeg', 'tiff'], 
            key="image",
            help="Upload the UAV image for flood segmentation"
        )
    
    with col2:
        st.subheader("🎯 Ground Truth Mask (Optional)")
        uploaded_mask = st.file_uploader(
            "Choose ground truth mask", 
            type=['png', 'jpg', 'jpeg', 'tiff'], 
            key="mask",
            help="Optional: Upload ground truth for evaluation metrics"
        )
    
    if uploaded_image is not None:
        # Load and display original image
        image = Image.open(uploaded_image)
        image_array = np.array(image)
        
        st.header("🖼️ Input Images")
        
        if uploaded_mask is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original UAV Image", use_container_width=True)
            with col2:
                mask = Image.open(uploaded_mask)
                mask_array = np.array(mask)
                st.image(mask, caption="Ground Truth Mask", use_container_width=True)
        else:
            st.image(image, caption="Original UAV Image", use_container_width=True)
            mask_array = None
        
# Process button
        if st.button("🚀 Start Segmentation", type="primary", use_container_width=True):
            with st.spinner("Processing flood segmentation..."):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Find optimal K
                status_text.text("Finding optimal K value...")
                progress_bar.progress(20)
                
                if use_auto_k:
                    if mask_array is not None:
                        best_k, scores = find_best_k(image_array, mask_array, (k_range_min, k_range_max))
                        st.success(f"✅ Best K found using F1-Score: {best_k}")
                        
                        # Plot K vs Score
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(range(k_range_min, k_range_max + 1), scores, 'bo-', linewidth=2, markersize=8)
                        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
                        ax.set_xlabel('K Value', fontsize=12)
                        ax.set_ylabel('F1-Score', fontsize=12)
                        ax.set_title('K Value Optimization using F1-Score', fontsize=14, fontweight='bold')
                        ax.legend(fontsize=12)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                    else:
                        best_k, inertias = find_best_k(image_array, None, (k_range_min, k_range_max))
                        st.success(f"✅ Best K found using elbow method: {best_k}")
                        
                        # Plot elbow curve
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(range(k_range_min, k_range_max + 1), inertias, 'bo-', linewidth=2, markersize=8)
                        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K = {best_k}')
                        ax.set_xlabel('K Value', fontsize=12)
                        ax.set_ylabel('Inertia', fontsize=12)
                        ax.set_title('Elbow Method for Optimal K Selection', fontsize=14, fontweight='bold')
                        ax.legend(fontsize=12)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                    
                    k_to_use = best_k
                else:
                    k_to_use = k_value
                    st.info(f"Using manual K value: {k_to_use}")
                
                # Step 2: Apply segmentation
                status_text.text("Applying K-means clustering...")
                progress_bar.progress(50)
                
                segmented_img, labels = apply_kmeans(image_array, k_to_use)

                # Store results in session state
                st.session_state["labels"] = labels
                st.session_state["segmented_img"] = segmented_img
                st.session_state["k_used"] = k_to_use
                st.session_state["segmentation_done"] = True
                st.session_state["image_array"] = image_array
                st.session_state["pixel_size_m2"] = pixel_size_m2
                st.session_state["overlay_alpha"] = overlay_alpha
                st.session_state["mask_array"] = mask_array
                st.session_state["uploaded_image_name"] = uploaded_image.name
                
                progress_bar.progress(100)
                status_text.text("✅ Segmentation complete! Please select flood label below.")
                
                st.success("✅ Segmentasi selesai! Pilih label banjir di bawah.")

        # Check if segmentation is done and show flood label selection
        if st.session_state.get("segmentation_done", False):
            st.header("🏷️ Flood Label Selection")
            
            # Get data from session state
            labels = st.session_state["labels"]
            segmented_img = st.session_state["segmented_img"]
            k_used = st.session_state["k_used"]
            image_array = st.session_state["image_array"]
            pixel_size_m2 = st.session_state["pixel_size_m2"]
            overlay_alpha = st.session_state["overlay_alpha"]
            mask_array = st.session_state.get("mask_array")
            uploaded_image_name = st.session_state["uploaded_image_name"]
            
            # Show individual cluster visualizations
            st.subheader("🎨 Individual Cluster Visualization")
            st.info("💡 **Tip:** Lihat setiap cluster secara terpisah untuk mengidentifikasi mana yang merepresentasikan area banjir (biasanya area berwarna coklat/keruh)")
            
            unique_labels = np.unique(labels)
            
            # Create individual cluster masks for better visualization
            cols = st.columns(min(4, len(unique_labels)))
            
            for idx, label in enumerate(unique_labels):
                col_idx = idx % 4
                with cols[col_idx]:
                    # Create binary mask for this cluster
                    cluster_mask = (labels == label).astype(np.uint8) * 255
                    
                    # Create colored overlay of original image with cluster highlighted
                    cluster_overlay = image_array.copy()
                    
                    # Create colored mask overlay
                    colored_mask = np.zeros_like(image_array)
                    colored_mask[labels == label] = [255, 0, 0]  # Red highlight
                    
                    # Blend with original image
                    cluster_highlighted = cv2.addWeighted(cluster_overlay, 0.7, colored_mask, 0.3, 0)
                    
                    pixel_count = np.sum(labels == label)
                    percentage = (pixel_count / labels.size) * 100
                    
                    st.image(cluster_highlighted, 
                            caption=f"Cluster {label}\n{pixel_count:,} pixels ({percentage:.1f}%)", 
                            use_container_width=True)
            
            # Show additional reference images
            st.subheader("📋 Reference Images")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(image_array, caption="Original Image", use_container_width=True)
            
            with col2:
                st.image(segmented_img, caption=f"K-means Segmentation (K={k_used})", use_container_width=True)
            
            with col3:
                # Create cluster labels visualization with different colors
                colored_labels = plt.cm.tab10(labels / labels.max())[:, :, :3]
                colored_labels = (colored_labels * 255).astype(np.uint8)
                st.image(colored_labels, caption="Cluster Labels (Colored)", use_container_width=True)
            
            # Enhanced flood label selection with better guidance
            st.subheader("🌊 Select Flood Cluster")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                unique_labels = np.unique(labels)
                flood_label = st.selectbox(
                    "Pilih cluster yang menunjukkan area banjir:",
                    unique_labels,
                    key="flood_label_selector",
                    help="💡 Pilih cluster yang menunjukkan area berwarna coklat/keruh (air banjir) berdasarkan visualisasi di atas"
                )
                
                # Show preview of selected cluster
                if flood_label is not None:
                    selected_mask = (labels == flood_label).astype(np.uint8) * 255
                    selected_overlay = image_array.copy()
                    colored_mask = np.zeros_like(image_array)
                    colored_mask[labels == flood_label] = [0, 255, 255]  # Yellow highlight for preview
                    preview_image = cv2.addWeighted(selected_overlay, 0.6, colored_mask, 0.4, 0)
                    
                    st.image(preview_image, 
                            caption=f"Preview: Cluster {flood_label} (Highlighted in Yellow)", 
                            use_container_width=True)
            
            with col2:
                st.subheader("📊 Cluster Statistics")
                for label in unique_labels:
                    pixel_count = np.sum(labels == label)
                    percentage = (pixel_count / labels.size) * 100
                    area_m2 = pixel_count * pixel_size_m2
                    
                    # Highlight selected cluster
                    if label == flood_label:
                        st.markdown(f"""
                        **🌊 Cluster {label} (SELECTED)**
                        - Pixels: {pixel_count:,}
                        - Percentage: {percentage:.1f}%
                        - Area: {area_m2:.1f} m²
                        """)
                    else:
                        st.markdown(f"""
                        **Cluster {label}**
                        - Pixels: {pixel_count:,}
                        - Percentage: {percentage:.1f}%
                        - Area: {area_m2:.1f} m²
                        """)
                
                st.markdown("""
                **🔍 Tips Memilih Cluster Banjir:**
                - Cari area berwarna coklat/keruh
                - Biasanya area yang luas dan kontinu
                - Hindari vegetasi (hijau) atau bangunan
                - Perhatikan area yang terlihat seperti air
                """)
            
            # Button to process flood analysis
            if st.button("🌊 Analyze Flood Areas", type="primary", use_container_width=True):
                with st.spinner("Analyzing flood areas..."):
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 3: Calculate flood area
                    status_text.text("Calculating flood area...")
                    progress_bar.progress(30)
                    
                    flood_area_m2, flood_pixels, _ = calculate_flood_area(labels, flood_label, pixel_size_m2)
                    
                    # Step 4: Create visualizations
                    status_text.text("Creating visualizations...")
                    progress_bar.progress(60)
                    
                    # Create flood overlay
                    flood_overlay = create_flood_overlay(image_array, labels, flood_label, overlay_alpha)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Store flood analysis results
                    st.session_state["flood_analysis_done"] = True
                    st.session_state["flood_label"] = flood_label
                    st.session_state["flood_area_m2"] = flood_area_m2
                    st.session_state["flood_pixels"] = flood_pixels
                    st.session_state["flood_overlay"] = flood_overlay
                    
                    st.success("✅ Analisis banjir selesai!")

        # Display results if flood analysis is done
        if st.session_state.get("flood_analysis_done", False):
            # Get all necessary data from session state
            labels = st.session_state["labels"]
            segmented_img = st.session_state["segmented_img"]
            k_used = st.session_state["k_used"]
            image_array = st.session_state["image_array"]
            pixel_size_m2 = st.session_state["pixel_size_m2"]
            mask_array = st.session_state.get("mask_array")
            uploaded_image_name = st.session_state["uploaded_image_name"]
            flood_label = st.session_state["flood_label"]
            flood_area_m2 = st.session_state["flood_area_m2"]
            flood_pixels = st.session_state["flood_pixels"]
            flood_overlay = st.session_state["flood_overlay"]
            
            # Display results
            st.header("📊 Segmentation Results")
            
            # Main visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎨 Segmented Image")
                st.image(segmented_img, caption=f"K-means Segmentation (K={k_used})", use_container_width=True)
            
            with col2:
                st.subheader("🌊 Flood Areas Overlay")
                st.image(flood_overlay, caption="Flood areas highlighted in red", use_container_width=True)
            
            # Cluster visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cluster labels
            im1 = ax1.imshow(labels, cmap='tab10')
            ax1.set_title(f'Cluster Labels (K={k_used})', fontsize=14, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # Flood mask
            flood_mask = (labels == flood_label).astype(int)
            im2 = ax2.imshow(flood_mask, cmap='Blues')
            ax2.set_title('Flood Areas (Binary Mask)', fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Metrics display
            st.header("📈 Analysis Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌊 Flood Area", f"{flood_area_m2:.2f} m²")
            with col2:
                st.metric("📍 Flood Pixels", f"{flood_pixels:,}")
            with col3:
                total_pixels = labels.size
                flood_percentage = (flood_pixels / total_pixels) * 100
                st.metric("📊 Coverage", f"{flood_percentage:.2f}%")
            with col4:
                st.metric("🎯 K Value Used", f"{k_used}")
            
            # Detailed cluster information
            st.subheader("🔍 Cluster Analysis")
            cluster_data = []
            colors = plt.cm.tab10(np.linspace(0, 1, k_used))
            
            for i in range(k_used):
                cluster_pixels = np.sum(labels == i)
                cluster_percentage = (cluster_pixels / labels.size) * 100
                cluster_area = cluster_pixels * pixel_size_m2
                is_flood = i == flood_label 
                
                cluster_data.append({
                    'Cluster': i,
                    'Flood Area': "🌊 Yes" if is_flood else "🏞️ No",
                    'Pixels': f"{cluster_pixels:,}",
                    'Percentage': f"{cluster_percentage:.2f}%",
                    'Area (m²)': f"{cluster_area:.2f}"
                })
            
            st.table(cluster_data)
            
            # Evaluation metrics if ground truth is available
            if mask_array is not None:
                st.header("🎯 Evaluation Metrics")
                
                binary_mask = binarize_mask(mask_array)
                resized_mask = cv2.resize(binary_mask, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_NEAREST)
                metrics, pred_mask = evaluate_segmentation(labels, resized_mask, k_used)
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.3f}")
                    st.metric("✅ True Positives", f"{metrics['TP']:,}")
                with col2:
                    st.metric("🎪 Precision", f"{metrics['Precision']:.3f}")
                    st.metric("❌ False Positives", f"{metrics['FP']:,}")
                with col3:
                    st.metric("🔄 Recall", f"{metrics['Recall']:.3f}")
                    st.metric("✅ True Negatives", f"{metrics['TN']:,}")
                with col4:
                    st.metric("🏆 F1-Score", f"{metrics['F1-Score']:.3f}")
                    st.metric("❌ False Negatives", f"{metrics['FN']:,}")
                
                # Confusion matrix visualization
                st.subheader("📊 Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                conf_matrix = np.array([[metrics['TN'], metrics['FP']], 
                                      [metrics['FN'], metrics['TP']]])
                im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                # Labels and annotations
                classes = ['Non-Flood', 'Flood']
                tick_marks = np.arange(len(classes))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                
                # Add text annotations
                thresh = conf_matrix.max() / 2.
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, format(conf_matrix[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if conf_matrix[i, j] > thresh else "black",
                               fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Side-by-side comparison
                st.subheader("🔍 Prediction vs Ground Truth Comparison")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(pred_mask, caption="Predicted Flood Mask", use_container_width=True, clamp=True)
                with col2:
                    st.image(resized_mask, caption="Ground Truth Mask", use_container_width=True, clamp=True)
                with col3:
                    # Create difference image
                    diff_mask = np.abs(pred_mask.astype(int) - resized_mask.astype(int))
                    st.image(diff_mask, caption="Difference (Errors)", use_container_width=True, clamp=True)
            
            # Download section
            st.header("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download segmented image
                segmented_pil = Image.fromarray(segmented_img)
                buf = io.BytesIO()
                segmented_pil.save(buf, format='PNG')
                st.download_button(
                    label="📥 Segmented Image",
                    data=buf.getvalue(),
                    file_name=f"segmented_k{k_used}_{uploaded_image_name}",
                    mime="image/png"
                )
            
            with col2:
                # Download flood overlay
                overlay_pil = Image.fromarray(flood_overlay)
                buf = io.BytesIO()
                overlay_pil.save(buf, format='PNG')
                st.download_button(
                    label="📥 Flood Overlay",
                    data=buf.getvalue(),
                    file_name=f"flood_overlay_{uploaded_image_name}",
                    mime="image/png"
                )
            
            with col3:
                # Download cluster labels
                labels_normalized = ((labels / labels.max()) * 255).astype(np.uint8)
                labels_pil = Image.fromarray(labels_normalized)
                buf = io.BytesIO()
                labels_pil.save(buf, format='PNG')
                st.download_button(
                    label="📥 Cluster Labels",
                    data=buf.getvalue(),
                    file_name=f"labels_k{k_used}_{uploaded_image_name}",
                    mime="image/png"
                )
            
            # Summary report
            st.header("📋 Summary Report")
            
            summary_text = f"""
## Flood Segmentation Analysis Report

**Image:** {uploaded_image_name}
**Processing Date:** {st.session_state.get('processing_date', 'N/A')}

### Parameters Used
- K-means clusters: {k_used}
- Selected flood label: {flood_label}
- Pixel size: {pixel_size_m2} m²/pixel

### Results
- **Total flood area:** {flood_area_m2:.2f} m²
- **Flood coverage:** {(flood_pixels / labels.size) * 100:.2f}% of total area
- **Flood pixels:** {flood_pixels:,} pixels
- **Total image pixels:** {labels.size:,} pixels

### Cluster Distribution
"""
            for i, cluster_info in enumerate(cluster_data):
                summary_text += f"- **Cluster {i}:** {cluster_info['Area (m²)']} m² ({cluster_info['Percentage']})\n"
            
            if mask_array is not None:
                summary_text += f"""
### Evaluation Metrics (vs Ground Truth)
- **Accuracy:** {metrics['Accuracy']:.3f}
- **Precision:** {metrics['Precision']:.3f}
- **Recall:** {metrics['Recall']:.3f}
- **F1-Score:** {metrics['F1-Score']:.3f}
- **True Positives:** {metrics['TP']:,}
- **False Positives:** {metrics['FP']:,}
- **True Negatives:** {metrics['TN']:,}
- **False Negatives:** {metrics['FN']:,}
"""
            
            st.markdown(summary_text)
            
            # Download summary report
            st.download_button(
                label="📄 Download Summary Report",
                data=summary_text,
                file_name=f"flood_analysis_report_{uploaded_image_name.split('.')[0]}.md",
                mime="text/markdown"
            )
            
            # Add button to reset and start over
            if st.button("🔄 Start New Analysis", type="secondary"):
                # Clear session state
                for key in ["segmentation_done", "flood_analysis_done", "labels", "segmented_img", 
                           "k_used", "image_array", "pixel_size_m2", "overlay_alpha", "mask_array",
                           "uploaded_image_name", "flood_label", "flood_area_m2", "flood_pixels", "flood_overlay"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # Footer with instructions
    if uploaded_image is None:
        st.info("👆 Please upload a UAV image to begin flood segmentation analysis.")
        
        st.markdown("""
        ### 📖 How to Use
        1. **Upload UAV Image**: Select your drone/UAV image file
        2. **Optional**: Upload ground truth mask for evaluation
        3. **Configure Parameters**: 
        - Enable auto K-selection or choose manually
        - Set pixel size for accurate area calculation
        4. **Process**: Click the segmentation button
        5. **Select Flood Label**: Choose which cluster represents flood areas
        6. **Analyze Results**: View flood areas, metrics, and download results
        
        ### 🎯 Supported Features
        - **Automatic K-value optimization** using F1-score or elbow method
        - **Interactive flood label selection** with visual reference
        - **Flood area calculation** in square meters
        - **Performance evaluation** with ground truth masks
        - **Multiple visualization options** with overlays
        - **Downloadable results** in various formats
        """)

if __name__ == "__main__":
    main()