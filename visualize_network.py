import matplotlib.pyplot as plt
import numpy as np

def draw_network(filename="neural_network_architecture.png"):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Layer titles and names
    layer_names = [
        "Input Layer\n(27 States)", 
        "Hidden Layer 1\n(64 Nodes, ReLU)", 
        "Hidden Layer 2\n(64 Nodes, ReLU)", 
        "Output Layer\n(7 Actions, Linear)"
    ]
    
    x_positions = [0, 3, 6, 9]
    
    # Node Y-coordinates for plotting (simplified representation of wide layers)
    # Input, Hidden 1, Hidden 2 will have 8 nodes drawn: 4 at top, 4 at bottom, with a '...' gap
    nodes_y = {
        0: [8, 7, 6, 5, 3, 2, 1, 0],  # 8 nodes representation
        1: [8, 7, 6, 5, 3, 2, 1, 0],  # 8 nodes representation
        2: [8, 7, 6, 5, 3, 2, 1, 0],  # 8 nodes representation
        3: [7.5, 6.25, 5.0, 3.75, 2.5, 1.25, 0.0]  # 7 actions (all drawn explicitly)
    }
    
    actions_labels = [
        "Do nothing",
        "Water Pump ON",
        "Water Pump OFF",
        "Heater ON",
        "Heater OFF",
        "Open shade curtains",
        "Close shade curtains"
    ]
    
    # Colors
    color_input = '#3498db'    # Teal-blue for inputs
    color_hidden = '#9b59b6'   # Deep purple for hidden representations
    color_output = '#2ecc71'   # Green for action values
    
    # 1. Draw connections first so they lie behind nodes
    for l in range(3):
        x1 = x_positions[l]
        x2 = x_positions[l+1]
        y1s = nodes_y[l]
        y2s = nodes_y[l+1]
        
        # Connect each drawn node in layer l to every drawn node in layer l+1
        for i1, y1 in enumerate(y1s):
            # Skip the index that acts as a visual placeholder for '...'
            if l < 3 and i1 == 4:
                continue
            for i2, y2 in enumerate(y2s):
                if l+1 < 3 and i2 == 4:
                    continue
                ax.plot([x1, x2], [y1, y2], color='#bdc3c7', alpha=0.18, linewidth=0.9, zorder=1)
                
    # 2. Draw nodes and label them
    for l in range(4):
        x = x_positions[l]
        ys = nodes_y[l]
        
        # Draw Layer Title Header
        ax.text(x, 9.2, layer_names[l], ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
        
        # Select layer specific node color
        if l == 0:
            color = color_input
        elif l in [1, 2]:
            color = color_hidden
        else:
            color = color_output
            
        for i, y in enumerate(ys):
            # Draw ellipsis in the middle of standard layers
            if l < 3 and i == 4:
                ax.text(x, 4.0, "⋮", ha='center', va='center', fontsize=20, fontweight='bold', color='#7f8c8d', zorder=2)
                continue
            
            # Draw node circle
            circle = plt.Circle((x, y), 0.18, color=color, ec='#2c3e50', lw=1.5, zorder=3)
            ax.add_patch(circle)
            
            # Add labels based on layer
            if l == 0:
                # Input Labels (Greenhouse State descriptions)
                state_labels = {
                    0: "State 1\n(Dry, Low, Low)",
                    1: "State 2\n(Dry, Low, Med)",
                    2: "State 3\n(Dry, Low, High)",
                    3: "State 4\n(Dry, Opt, Low)",
                    5: "State 25\n(Wet, High, Low)",
                    6: "State 26\n(Wet, High, Med)",
                    7: "State 27\n(Wet, High, High)"
                }
                label_text = state_labels.get(i, "")
                if label_text:
                    ax.text(x - 0.3, y, label_text, ha='right', va='center', fontsize=9, color='#34495e', fontweight='medium')
                    
            elif l == 3:
                # Output Labels (Q-Values of actions)
                label_text = f"Q(s, {actions_labels[i]})"
                ax.text(x + 0.3, y, label_text, ha='left', va='center', fontsize=10, color='#2c3e50', fontweight='bold')
                
            else:
                # Hidden Layer Nodes
                node_idx = i + 1 if i < 4 else i + 57
                label_text = f"$h^{{({l})}}_{{{node_idx}}}$"
                ax.text(x + 0.22, y + 0.15, label_text, ha='left', va='bottom', fontsize=8, color='#7f8c8d', zorder=4)
                
    # Add titles and parameters summary
    fig.text(0.5, 0.96, "Greenhouse DQN Modular Neural Network Architecture", ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
    
    param_text = (
        "Total Trainable Parameters:\n"
        "Layer 1 (fc1): 27 inputs × 64 outputs + 64 biases = 1,792 params\n"
        "Layer 2 (fc2): 64 inputs × 64 outputs + 64 biases = 4,160 params\n"
        "Layer 3 (out): 64 inputs × 7 outputs + 7 biases = 455 params\n"
        "Total: 6,407 Parameters"
    )
    fig.text(0.5, -0.02, param_text, ha='center', va='center', fontsize=10, color='#2c3e50', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network architecture diagram successfully saved to '{filename}'!")

if __name__ == '__main__':
    draw_network()
