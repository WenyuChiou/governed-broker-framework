"""
Generate Skill-Governed Framework Architecture Diagram v0.3

Creates a professional architecture diagram using matplotlib.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    config_color = '#E8E8E8'  # Light gray
    llm_color = '#D4E6F1'     # Light blue
    broker_color = '#E8DAEF'  # Light purple
    sim_color = '#D5F5E3'     # Light green
    box_color = '#FFFFFF'     # White boxes
    
    # Layer 1: Configuration (Top)
    ax.add_patch(FancyBboxPatch((0.5, 9.5), 15, 1.8, boxstyle="round,pad=0.1", 
                                 facecolor=config_color, edgecolor='gray', linewidth=2))
    ax.text(8, 11, 'CONFIGURATION LAYER', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # YAML files
    for i, (name, x) in enumerate([('providers.yaml', 3), ('domain.yaml', 8), ('skill_registry.yaml', 13)]):
        ax.add_patch(FancyBboxPatch((x-1.2, 9.7), 2.4, 1, boxstyle="round,pad=0.05", 
                                     facecolor='#FFF9C4', edgecolor='#FBC02D', linewidth=1.5))
        ax.text(x, 10.2, name, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Layer 2: LLM Providers
    ax.add_patch(FancyBboxPatch((0.5, 6.8), 15, 2.4, boxstyle="round,pad=0.1", 
                                 facecolor=llm_color, edgecolor='#2980B9', linewidth=2))
    ax.text(8, 8.9, 'LLM PROVIDER LAYER', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # LLMProviderRegistry
    ax.add_patch(FancyBboxPatch((1, 7.5), 6, 1.2, boxstyle="round,pad=0.05", 
                                 facecolor=box_color, edgecolor='#2980B9', linewidth=1.5))
    ax.text(4, 8.4, 'LLMProviderRegistry', ha='center', va='center', fontsize=10, fontweight='bold')
    
    for i, name in enumerate(['OllamaProvider', 'OpenAIProvider']):
        ax.add_patch(FancyBboxPatch((1.3 + i*2.5, 7.6), 2.3, 0.6, boxstyle="round,pad=0.03", 
                                     facecolor='#FFFFFF', edgecolor='gray', linewidth=1))
        ax.text(2.45 + i*2.5, 7.9, name, ha='center', va='center', fontsize=8)
    
    # Support components
    for i, name in enumerate(['RateLimiter', 'AsyncAdapter', 'RetryHandler']):
        ax.add_patch(FancyBboxPatch((8 + i*2.5, 7.1), 2.3, 0.8, boxstyle="round,pad=0.03", 
                                     facecolor=box_color, edgecolor='#2980B9', linewidth=1))
        ax.text(9.15 + i*2.5, 7.5, name, ha='center', va='center', fontsize=8)
    
    # Layer 3: Governed Broker
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 15, 4, boxstyle="round,pad=0.1", 
                                 facecolor=broker_color, edgecolor='#8E44AD', linewidth=2))
    ax.text(8, 6.2, 'GOVERNED BROKER LAYER', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Top row components
    components = [('DomainConfigLoader', 2), ('ModelAdapter', 5.5), ('SkillProposal', 9), ('SkillRegistry', 12.5)]
    for name, x in components:
        ax.add_patch(FancyBboxPatch((x-1.3, 5.3), 2.6, 0.7, boxstyle="round,pad=0.03", 
                                     facecolor=box_color, edgecolor='#8E44AD', linewidth=1))
        ax.text(x, 5.65, name, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows between components
    for i in range(len(components)-1):
        ax.annotate('', xy=(components[i+1][1]-1.3, 5.65), xytext=(components[i][1]+1.3, 5.65),
                   arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.5))
    
    # Validation Pipeline
    ax.add_patch(FancyBboxPatch((1, 3.3), 14, 1.6, boxstyle="round,pad=0.05", 
                                 facecolor='#F5EEF8', edgecolor='#8E44AD', linewidth=1.5))
    ax.text(8, 4.7, 'VALIDATION PIPELINE', ha='center', va='center', fontsize=10, fontweight='bold')
    
    validators = ['Admissibility', 'Feasibility', 'Constraints', 'EffectSafety', 'PMT', 'Uncertainty']
    for i, name in enumerate(validators):
        x = 2 + i * 2.2
        ax.add_patch(FancyBboxPatch((x-0.9, 3.5), 1.8, 0.6, boxstyle="round,pad=0.02", 
                                     facecolor=box_color, edgecolor='#AF7AC5', linewidth=1))
        ax.text(x, 3.8, name, ha='center', va='center', fontsize=7)
        if i < len(validators) - 1:
            ax.annotate('', xy=(x+1.1, 3.8), xytext=(x+0.9, 3.8),
                       arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1))
    
    # Bottom row
    ax.add_patch(FancyBboxPatch((2, 2.7), 3, 0.5, boxstyle="round,pad=0.02", 
                                 facecolor=box_color, edgecolor='#8E44AD', linewidth=1))
    ax.text(3.5, 2.95, 'ValidatorFactory', ha='center', va='center', fontsize=8)
    
    ax.add_patch(FancyBboxPatch((11, 2.7), 3, 0.5, boxstyle="round,pad=0.02", 
                                 facecolor=box_color, edgecolor='#8E44AD', linewidth=1))
    ax.text(12.5, 2.95, 'AuditWriter', ha='center', va='center', fontsize=8)
    
    # Layer 4: Simulation
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 15, 1.7, boxstyle="round,pad=0.1", 
                                 facecolor=sim_color, edgecolor='#27AE60', linewidth=2))
    ax.text(8, 1.9, 'SIMULATION / WORLD LAYER', ha='center', va='center', fontsize=12, fontweight='bold')
    
    for i, name in enumerate(['SimulationEngine', 'Agent State', 'Environment']):
        ax.add_patch(FancyBboxPatch((2 + i*4.5, 0.7), 3.5, 0.7, boxstyle="round,pad=0.03", 
                                     facecolor=box_color, edgecolor='#27AE60', linewidth=1))
        ax.text(3.75 + i*4.5, 1.05, name, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Main flow arrow
    ax.annotate('', xy=(8, 2.5), xytext=(8, 6.8),
               arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.text(8.3, 4.5, 'Approved\nSkill', ha='left', va='center', fontsize=8, style='italic')
    
    # Title
    ax.text(8, 11.7, 'Skill-Governed Framework v0.3 Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('skill_architecture_v03.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: skill_architecture_v03.png")
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()
