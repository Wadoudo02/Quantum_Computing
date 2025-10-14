"""
Phase 2: Bell's Inequality Interactive App
===========================================

Interactive Streamlit application for exploring entanglement and
demonstrating quantum non-locality through Bell's inequality violation.

Run with:
    streamlit run app.py

Features:
- Interactive Bell state explorer
- Real-time CHSH inequality demonstration
- Correlation measurement visualization
- High-quality plot export for presentations
- Animated demonstrations

Reference: Based on the section on Bell's inequality and entanglement
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase2_entanglement.bell_states import (
    BellState, bell_phi_plus, bell_phi_minus,
    bell_psi_plus, bell_psi_minus, create_bell_state
)
from phase2_entanglement.bells_inequality import (
    compute_chsh_value, demonstrate_bell_violation,
    optimal_chsh_angles, classical_bound, quantum_bound,
    measure_correlation, scan_chsh_angles, explain_bell_violation
)
from phase2_entanglement.visualization import (
    plot_density_matrix, plot_chsh_demonstration,
    plot_entanglement_comparison, create_linkedin_summary
)

# Page config
st.set_page_config(
    page_title="Phase 2: Bell's Inequality",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create plots directory
PLOTS_DIR = Path(__file__).parent.parent.parent.parent / "plots" / "phase2"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("🔔 Phase 2: Entanglement")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode:",
    [
        "🏠 Overview",
        "🔔 Bell States Explorer",
        "📊 CHSH Inequality Demo",
        "📈 Correlation Measurements",
        "🎬 Animations",
        "📚 Theory & Explanation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This app demonstrates **Bell's inequality** - one of the most profound
discoveries in quantum mechanics.

It proves that quantum entanglement exhibits non-local correlations that
cannot be explained by any local hidden variable theory.
""")


# ============================================================================
# Mode: Overview
# ============================================================================

if mode == "🏠 Overview":
    st.title("🔔 Phase 2: Entanglement and Bell's Inequality")

    st.markdown("""
    ## Welcome to Quantum Non-Locality!

    This interactive application explores **Bell's inequality** and demonstrates
    that quantum mechanics exhibits **non-local correlations** that violate
    classical physics.

    ### What You'll Learn:

    1. **Bell States** - The four maximally entangled two-qubit states
    2. **CHSH Inequality** - A mathematical test of local realism
    3. **Quantum Violation** - How quantum mechanics exceeds classical bounds
    4. **Non-Locality** - What "spooky action at a distance" really means

    ### The Key Result:

    For any classical (local hidden variable) theory:
    """)

    st.latex(r"|E(a,b) + E(a,b') + E(a',b) - E(a',b')| \leq 2")

    st.markdown("""
    But quantum mechanics can achieve:
    """)

    st.latex(r"|S| = 2\sqrt{2} \approx 2.828")

    st.success("""
    **This violation proves that quantum entanglement is fundamentally different
    from any classical correlation!**
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Classical Bound",
            value="2.000",
            delta=None
        )
        st.caption("Maximum for local theories")

    with col2:
        st.metric(
            label="Quantum Prediction",
            value="2.828",
            delta="+0.828",
            delta_color="normal"
        )
        st.caption("Quantum mechanics violates!")

    with col3:
        st.metric(
            label="Violation",
            value="41.4%",
            delta="+41.4%",
            delta_color="normal"
        )
        st.caption("Above classical limit")

    st.markdown("---")

    st.markdown("### 🎯 Quick Start Guide:")
    st.markdown("""
    - **Bell States Explorer**: See all four Bell states and their properties
    - **CHSH Demo**: Run the actual inequality test and see violation in real-time
    - **Correlations**: Measure quantum correlations at different angles
    - **Animations**: Create LinkedIn-ready visualizations
    - **Theory**: Deep dive into the mathematics and meaning
    """)


# ============================================================================
# Mode: Bell States Explorer
# ============================================================================

elif mode == "🔔 Bell States Explorer":
    st.title("🔔 Bell States Explorer")

    st.markdown("""
    The **Bell states** are the four maximally entangled two-qubit states.
    They form a complete orthonormal basis for the two-qubit Hilbert space.
    """)

    # Select state
    state_choice = st.selectbox(
        "Choose a Bell state:",
        ["Φ+ (Phi Plus)", "Φ- (Phi Minus)", "Ψ+ (Psi Plus)", "Ψ- (Psi Minus)"]
    )

    state_map = {
        "Φ+ (Phi Plus)": bell_phi_plus(),
        "Φ- (Phi Minus)": bell_phi_minus(),
        "Ψ+ (Psi Plus)": bell_psi_plus(),
        "Ψ- (Psi Minus)": bell_psi_minus()
    }

    state = state_map[state_choice]

    # Display state information
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### State Vector")
        st.latex(str(state))

        st.markdown("### Properties")
        entropy = state.von_neumann_entropy()
        is_entangled = state.is_entangled()
        schmidt_coeffs, _, _ = state.schmidt_decomposition()

        st.metric("Entangled", "Yes" if is_entangled else "No")
        st.metric("Von Neumann Entropy", f"{entropy:.4f}")
        st.metric("Schmidt Rank", "2" if is_entangled else "1")

        st.markdown("### Schmidt Coefficients")
        st.write(f"λ₁ = {schmidt_coeffs[0]:.4f}")
        st.write(f"λ₂ = {schmidt_coeffs[1]:.4f}")

    with col2:
        st.markdown("### Density Matrix")

        fig = plot_density_matrix(state)
        st.pyplot(fig)
        plt.close()

        if st.button("💾 Save Density Matrix Plot", key="save_density"):
            save_path = PLOTS_DIR / f"density_matrix_{state.name}.png"
            plot_density_matrix(state, save_path=str(save_path))
            st.success(f"Saved to {save_path}")

    st.markdown("---")

    # Measurement simulation
    st.markdown("### 🎲 Measurement Simulation")

    n_shots = st.slider("Number of measurements:", 10, 10000, 1000, step=10)

    if st.button("🎲 Measure State"):
        results = state.measure(shots=n_shots)

        # Count outcomes
        outcomes_str = [''.join(map(str, r)) for r in results]
        unique, counts = np.unique(outcomes_str, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(unique, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(unique)],
               alpha=0.7, edgecolor='black')
        ax.set_xlabel('Measurement Outcome', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Measurement Results for |{state.name}⟩ ({n_shots} shots)', fontsize=14)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        for col, (outcome, count) in zip([col1, col2, col3, col4], zip(unique, counts)):
            col.metric(f"|{outcome}⟩", f"{count}", f"{count/n_shots*100:.1f}%")

    st.markdown("---")

    # Compare all Bell states
    if st.checkbox("📊 Compare All Bell States"):
        st.markdown("### All Four Bell States")

        fig = plot_entanglement_comparison()
        st.pyplot(fig)
        plt.close()

        if st.button("💾 Save Comparison Plot", key="save_comparison"):
            save_path = PLOTS_DIR / "bell_states_comparison.png"
            plot_entanglement_comparison(save_path=str(save_path))
            st.success(f"Saved to {save_path}")


# ============================================================================
# Mode: CHSH Inequality Demo
# ============================================================================

elif mode == "📊 CHSH Inequality Demo":
    st.title("📊 CHSH Inequality Demonstration")

    st.markdown("""
    ## The CHSH Test

    The CHSH inequality is a mathematical test that distinguishes quantum mechanics
    from classical (local hidden variable) theories.

    ### The Inequality:

    For measurements at angles *a*, *a'*, *b*, *b'*:
    """)

    st.latex(r"S = |E(a,b) + E(a,b') + E(a',b) - E(a',b')|")

    st.markdown("""
    - **Classical bound**: S ≤ 2
    - **Quantum maximum**: S ≤ 2√2 ≈ 2.828 (Tsirelson's bound)
    """)

    st.markdown("---")

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        shots_per_corr = st.slider(
            "Measurements per correlation:",
            1000, 20000, 10000, step=1000
        )

    with col2:
        num_trials = st.slider(
            "Number of independent trials:",
            10, 500, 100, step=10
        )

    use_optimal = st.checkbox("Use optimal angles (recommended)", value=True)

    if use_optimal:
        a, a_prime, b, b_prime = optimal_chsh_angles()
        st.info(f"""
        **Optimal angles for maximum violation:**
        - Alice's angles: a = {a:.3f}, a' = {a_prime:.3f} rad
        - Bob's angles: b = {b:.3f}, b' = {b_prime:.3f} rad
        """)
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            a = st.slider("Alice angle a:", 0.0, 2*np.pi, 0.0, 0.1)
        with col2:
            a_prime = st.slider("Alice angle a':", 0.0, 2*np.pi, np.pi/2, 0.1)
        with col3:
            b = st.slider("Bob angle b:", 0.0, 2*np.pi, np.pi/4, 0.1)
        with col4:
            b_prime = st.slider("Bob angle b':", 0.0, 2*np.pi, -np.pi/4, 0.1)

    st.markdown("---")

    # Run demonstration
    if st.button("🚀 Run CHSH Test", type="primary"):
        with st.spinner("Running Bell inequality test..."):
            results = demonstrate_bell_violation(shots=shots_per_corr, num_trials=num_trials)

        # Display results
        st.success("✅ Test Complete!")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mean CHSH Value",
                f"{results['mean_chsh']:.3f}",
                f"{results['mean_chsh'] - 2:.3f}"
            )

        with col2:
            st.metric(
                "Violation Rate",
                f"{results['violation_rate']*100:.1f}%"
            )

        with col3:
            st.metric(
                "Above Classical",
                f"{(results['mean_chsh']/2 - 1)*100:.1f}%"
            )

        with col4:
            violates = results['mean_chsh'] > 2.0
            st.metric(
                "Bell Violation?",
                "YES! 🎉" if violates else "No"
            )

        st.markdown("---")

        # Visualization
        st.markdown("### 📊 Results Visualization")

        fig = plot_chsh_demonstration(results)
        st.pyplot(fig)
        plt.close()

        if st.button("💾 Save Full Results", key="save_chsh"):
            save_path = PLOTS_DIR / "chsh_demonstration.png"
            plot_chsh_demonstration(results, save_path=str(save_path))
            st.success(f"Saved to {save_path}")

        # Interpretation
        st.markdown("### 🎯 Interpretation")

        if violates:
            st.success(f"""
            **✅ Bell's Inequality is VIOLATED!**

            Our measured value of S = {results['mean_chsh']:.3f} exceeds the classical
            bound of 2.000, proving that quantum entanglement exhibits non-local
            correlations that cannot be explained by any local hidden variable theory.

            This is direct experimental evidence for the "spooky action at a distance"
            that Einstein found so troubling!
            """)
        else:
            st.warning("""
            **Insufficient violation detected.**

            Try increasing the number of measurements or trials for better statistics.
            """)


# ============================================================================
# Mode: Correlation Measurements
# ============================================================================

elif mode == "📈 Correlation Measurements":
    st.title("📈 Quantum Correlation Measurements")

    st.markdown("""
    Explore how quantum correlations E(a,b) vary with measurement angles.

    For the Bell state |Φ+⟩, the correlation is:
    """)

    st.latex(r"E(a,b) = \cos(a - b)")

    st.markdown("---")

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        angle_a = st.slider(
            "Alice's measurement angle (degrees):",
            0, 360, 0, 5
        ) * np.pi / 180

    with col2:
        angle_b = st.slider(
            "Bob's measurement angle (degrees):",
            0, 360, 45, 5
        ) * np.pi / 180

    shots = st.slider("Number of measurements:", 1000, 20000, 5000, 1000)

    # Measure correlation
    state = bell_phi_plus()

    if st.button("📊 Measure Correlation"):
        # Exact prediction
        exact_corr = np.cos(angle_a - angle_b)

        # Simulated measurement
        measured_corr = measure_correlation(state, angle_a, angle_b, shots=shots)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Exact Prediction", f"{exact_corr:.4f}")

        with col2:
            st.metric("Measured Value", f"{measured_corr:.4f}")

        with col3:
            diff = abs(measured_corr - exact_corr)
            st.metric("Difference", f"{diff:.4f}")

        # Visualize
        st.markdown("### Correlation vs Angle Difference")

        angle_diffs = np.linspace(0, 2*np.pi, 100)
        correlations = np.cos(angle_diffs)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(angle_diffs * 180/np.pi, correlations, 'b-', linewidth=2, label='Exact: cos(θ)')
        ax.scatter([abs(angle_a - angle_b) * 180/np.pi], [measured_corr],
                  color='red', s=200, zorder=5, label=f'Measured: {measured_corr:.3f}')
        ax.scatter([abs(angle_a - angle_b) * 180/np.pi], [exact_corr],
                  color='green', s=150, marker='x', linewidths=3, zorder=5,
                  label=f'Exact: {exact_corr:.3f}')

        ax.set_xlabel('Angle Difference |a - b| (degrees)', fontsize=12)
        ax.set_ylabel('Correlation E(a,b)', fontsize=12)
        ax.set_title('Quantum Correlation vs Measurement Angle', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_ylim([-1.2, 1.2])

        st.pyplot(fig)
        plt.close()

    # Angle scan
    st.markdown("---")
    st.markdown("### 🔄 Scan Through All Angles")

    if st.button("🔄 Run Angle Scan"):
        scan_data = scan_chsh_angles(num_points=100)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(scan_data['angles'] * 180/np.pi, scan_data['chsh_values'],
               linewidth=3, color='purple', label='CHSH(θ)')
        ax.axhline(2.0, color='red', linestyle='--', linewidth=2,
                  label='Classical Bound', alpha=0.7)
        ax.axhline(quantum_bound(), color='blue', linestyle='--', linewidth=2,
                  label='Quantum Maximum', alpha=0.7)
        ax.fill_between(scan_data['angles'] * 180/np.pi, 2.0, 3.0,
                       alpha=0.2, color='green', label='Violation Region')

        ax.set_xlabel('Bob\'s Angle (degrees)', fontsize=12)
        ax.set_ylabel('CHSH Parameter S', fontsize=12)
        ax.set_title('CHSH Value vs Measurement Angle', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-3, 3])

        st.pyplot(fig)
        plt.close()


# ============================================================================
# Mode: Animations
# ============================================================================

elif mode == "🎬 Animations":
    st.title("🎬 Animations & LinkedIn Content")

    st.markdown("""
    Generate high-quality visualizations and animations perfect for
    sharing on LinkedIn or in presentations.
    """)

    st.markdown("---")

    # LinkedIn summary
    st.markdown("### 📱 LinkedIn Summary Image")

    st.markdown("""
    Generate a single, striking image that captures the key result of
    Bell's inequality violation - perfect for LinkedIn posts!
    """)

    if st.button("🎨 Generate LinkedIn Image", type="primary"):
        with st.spinner("Creating publication-quality figure..."):
            save_path = PLOTS_DIR / "bell_violation_linkedin.png"
            fig = create_linkedin_summary(save_path=str(save_path))
            st.pyplot(fig)
            plt.close()

        st.success(f"""
        ✅ **Image saved!**

        Location: `{save_path}`

        This image is optimized for LinkedIn (300 DPI, publication quality).

        **Suggested LinkedIn Post:**

        🔔 Just demonstrated Bell's Inequality violation using quantum entanglement!

        Quantum mechanics achieved S = 2.8+, exceeding the classical bound of 2.0.
        This proves that entangled particles exhibit genuine non-local correlations.

        🎯 Key Result: Quantum > Classical by 41%

        No local hidden variable theory can explain this. Einstein's "spooky action
        at a distance" is real! 👻

        #QuantumComputing #Physics #QuantumMechanics #BellsInequality
        """)

    st.markdown("---")

    # Comparison plots
    st.markdown("### 📊 Bell States Comparison")

    if st.button("🎨 Generate Bell States Comparison"):
        with st.spinner("Creating comparison figure..."):
            save_path = PLOTS_DIR / "bell_states_full_comparison.png"
            fig = plot_entanglement_comparison(save_path=str(save_path))
            st.pyplot(fig)
            plt.close()

        st.success(f"Saved to: {save_path}")

    st.markdown("---")

    # Animation (note: streamlit doesn't show animations directly, but we can save them)
    st.markdown("### 🎬 CHSH Violation Animation")

    st.markdown("""
    Create an animated GIF showing the Bell inequality violation building up
    over time as measurements accumulate.

    *Note: Animation is saved as a file - streamlit can't display animations directly.*
    """)

    if st.button("🎬 Generate Animation (GIF)"):
        st.warning("""
        **Note:** Generating animations takes time (30-60 seconds).

        The animation will be saved as a GIF file but won't display here.
        """)

        with st.spinner("Creating animation... this may take a minute..."):
            try:
                from phase2_entanglement.visualization import create_chsh_animation

                save_path = PLOTS_DIR / "chsh_violation_animation.gif"
                anim = create_chsh_animation(num_frames=50, save_path=str(save_path), fps=10)

                st.success(f"""
                ✅ **Animation created!**

                Location: `{save_path}`

                Upload this GIF to LinkedIn to show the violation appearing in real-time!
                """)
            except Exception as e:
                st.error(f"Animation generation failed: {e}")
                st.info("You may need to install ffmpeg or pillow for animation support.")


# ============================================================================
# Mode: Theory & Explanation
# ============================================================================

elif mode == "📚 Theory & Explanation":
    st.title("📚 Bell's Inequality: Theory & Explanation")

    st.markdown("""
    ## The Foundation of Quantum Non-Locality

    Bell's inequality is one of the most profound results in quantum mechanics.
    It proves that quantum entanglement cannot be explained by any theory based
    on local realism.
    """)

    st.markdown("---")

    # Historical context
    st.markdown("### 📜 Historical Context")

    st.markdown("""
    **1935**: Einstein, Podolsky, and Rosen (EPR) publish their famous paradox,
    arguing that quantum mechanics must be incomplete.

    **1964**: John Stewart Bell proves that any local hidden variable theory
    must satisfy certain inequalities.

    **1970s-present**: Experiments consistently violate Bell's inequalities,
    confirming quantum mechanics and ruling out local realism.
    """)

    # The mathematics
    st.markdown("---")
    st.markdown("### 📐 The Mathematics")

    st.markdown("""
    #### The CHSH Inequality

    For two observers (Alice and Bob) making measurements on an entangled pair:

    - Alice chooses between two measurement directions: **a** or **a'**
    - Bob chooses between two measurement directions: **b** or **b'**
    - Each measurement gives result +1 or -1

    Define the correlation:
    """)

    st.latex(r"E(a,b) = \langle A_a \cdot B_b \rangle")

    st.markdown("where the average is over many measurements.")

    st.markdown("The CHSH parameter is:")

    st.latex(r"S = |E(a,b) + E(a,b') + E(a',b) - E(a',b')|")

    st.markdown("""
    **Classical bound**: Any local hidden variable theory predicts S ≤ 2

    **Quantum prediction**: Quantum mechanics can achieve S = 2√2 ≈ 2.828

    **Experimental observation**: Real measurements confirm S > 2!
    """)

    # What it means
    st.markdown("---")
    st.markdown("### 💡 What It Means")

    tab1, tab2, tab3 = st.tabs(["Local Realism", "Quantum Mechanics", "The Violation"])

    with tab1:
        st.markdown("""
        #### Local Realism Explained

        **Local Realism** assumes two things:

        1. **Locality**: No influence can travel faster than light
           - Measuring particle A cannot instantly affect particle B

        2. **Realism**: Particles have definite properties before measurement
           - The outcome is determined by "hidden variables"

        Under these assumptions, correlations are limited by S ≤ 2.
        """)

    with tab2:
        st.markdown("""
        #### Quantum Mechanics Explained

        Quantum mechanics says:

        1. **Superposition**: Particles don't have definite properties until measured

        2. **Entanglement**: Measuring one particle instantly affects the other
           (not by sending a signal, but by collapsing the shared wavefunction)

        3. **Non-locality**: The system is fundamentally non-separable

        This allows S ≤ 2√2 ≈ 2.828 (Tsirelson's bound).
        """)

    with tab3:
        st.markdown("""
        #### The Violation Explained

        When we measure S > 2, we prove:

        ✅ **Quantum mechanics is correct**
        - The predictions match experiments

        ❌ **Local realism is false**
        - At least one assumption (locality or realism) must be wrong

        🎯 **Entanglement is real**
        - "Spooky action at a distance" is genuine

        🔬 **No hidden variables**
        - The universe is fundamentally probabilistic
        """)

    # Common questions
    st.markdown("---")
    st.markdown("### ❓ Common Questions")

    with st.expander("Does this allow faster-than-light communication?"):
        st.markdown("""
        **No!** While the correlations appear instantaneous, you cannot use them
        to send information faster than light.

        The measurement results are random - you only see the correlation when
        comparing notes, which requires classical communication.
        """)

    with st.expander("What is 'spooky action at a distance'?"):
        st.markdown("""
        Einstein's phrase for quantum entanglement. It means that measuring one
        particle instantly affects its entangled partner, regardless of distance.

        However, this doesn't violate relativity because no *information* travels
        faster than light.
        """)

    with st.expander("Why is the quantum limit 2√2 and not higher?"):
        st.markdown("""
        This is called **Tsirelson's bound**. It comes from the mathematical
        structure of quantum mechanics (Hilbert space, unitary operators).

        Interestingly, there could be theories ("superquantum" theories) that
        violate Bell's inequality even more, but our universe follows quantum
        mechanics with this specific bound.
        """)

    with st.expander("Has this been tested experimentally?"):
        st.markdown("""
        **Yes, many times!**

        Key experiments:
        - 1972: Freedman & Clauser (first experimental test)
        - 1982: Aspect's experiments (closing loopholes)
        - 2015: "Loophole-free" experiments (Nobel Prize 2022)

        All confirm: Quantum mechanics wins, local realism loses!
        """)

    # References
    st.markdown("---")
    st.markdown("### 📖 Further Reading")

    st.markdown("""
    - **Imperial College Notes**: Section on Bell's inequality
    - **Original Papers**:
      - J.S. Bell, "On the Einstein Podolsky Rosen Paradox" (1964)
      - CHSH: Clauser, Horne, Shimony, Holt (1969)
    - **Nobel Prize 2022**: Awarded to Aspect, Clauser, and Zeilinger for
      experimental tests of Bell's inequality
    """)


# ============================================================================
# Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Export Plots")

if st.sidebar.button("📁 Open Plots Folder"):
    st.sidebar.info(f"Plots are saved to:\n`{PLOTS_DIR}`")

st.sidebar.markdown("---")
st.sidebar.caption("""
**Phase 2: Entanglement**
Built with Streamlit
Based on Imperial College notes
""")
