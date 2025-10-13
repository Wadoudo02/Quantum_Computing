"""
Quantum Computing Interactive App
==================================

Streamlit app for exploring single-qubit quantum computing concepts.

Run with:
    streamlit run src/phase1_qubits/app.py

Or from project root:
    cd /path/to/Quantum_Computing
    python -m streamlit run src/phase1_qubits/app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_qubits.qubit import Qubit, ket_0, ket_1, ket_plus, ket_minus
from phase1_qubits.gates import (
    PAULI_X, PAULI_Y, PAULI_Z, HADAMARD, S_GATE, T_GATE,
    apply_gate, apply_sequence, rotation_x, rotation_y, rotation_z
)
from phase1_qubits.bloch_sphere import BlochSphere

# Page configuration
st.set_page_config(
    page_title="Quantum Computing Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def create_custom_qubit(theta, phi):
    """Create qubit from Bloch sphere angles."""
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return Qubit([alpha, beta], normalize=False)


def plot_bloch_matplotlib(qubit, title="Qubit State"):
    """Create Bloch sphere plot for Streamlit."""
    bloch = BlochSphere(figsize=(8, 8))
    bloch.add_qubit(qubit, label="ψ", color='red', show_vector=True)

    bloch._draw_sphere()
    bloch._draw_axes()
    bloch._draw_equator_and_meridians()
    bloch._draw_basis_states()
    bloch._plot_qubits()

    bloch.ax.set_xlabel('X', fontsize=12)
    bloch.ax.set_ylabel('Y', fontsize=12)
    bloch.ax.set_zlabel('Z', fontsize=12)
    bloch.ax.set_title(title, fontsize=16, fontweight='bold')
    bloch.ax.set_box_aspect([1, 1, 1])
    bloch.ax.view_init(elev=20, azim=45)
    bloch.ax.set_xlim([-1.3, 1.3])
    bloch.ax.set_ylim([-1.3, 1.3])
    bloch.ax.set_zlim([-1.3, 1.3])
    bloch.ax.grid(False)

    return bloch.fig


def plot_measurement_histogram(qubit, shots=1000):
    """Create histogram of measurement outcomes."""
    outcomes = qubit.measure(shots=shots)

    fig, ax = plt.subplots(figsize=(8, 5))

    counts = [np.sum(outcomes == 0), np.sum(outcomes == 1)]
    probs = [qubit.prob_0(), qubit.prob_1()]

    x = [0, 1]
    width = 0.35

    ax.bar([i - width/2 for i in x], counts, width, label='Measured', alpha=0.8, color='steelblue')
    ax.bar([i + width/2 for i in x], [p * shots for p in probs], width,
           label='Expected', alpha=0.8, color='orange')

    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Measurement Results ({shots} shots)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['|0⟩', '|1⟩'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    return fig


def main():
    st.markdown('<h1 class="main-header">⚛️ Quantum Computing Explorer</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Welcome!</b> This interactive app lets you explore single-qubit quantum computing.
    Create custom qubit states, apply gates, and visualize everything on the Bloch sphere.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for mode selection
    st.sidebar.title("🎮 Mode Selection")
    mode = st.sidebar.radio(
        "Choose a mode:",
        ["🎨 State Creator", "🔧 Gate Laboratory", "📊 Measurement Lab",
         "📚 Common States", "🎬 Gate Sequence"]
    )

    # ========================================================================
    # MODE 1: State Creator
    # ========================================================================
    if mode == "🎨 State Creator":
        st.markdown('<h2 class="sub-header">Create Custom Qubit States</h2>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Control Panel")

            # Angle sliders
            theta = st.slider(
                "θ (theta) - Polar angle",
                min_value=0.0,
                max_value=np.pi,
                value=np.pi/4,
                step=0.01,
                help="Controls latitude on Bloch sphere"
            )

            phi = st.slider(
                "φ (phi) - Azimuthal angle",
                min_value=0.0,
                max_value=2*np.pi,
                value=0.0,
                step=0.01,
                help="Controls longitude on Bloch sphere"
            )

            # Create qubit
            qubit = create_custom_qubit(theta, phi)

            # Display state information
            st.markdown("### State Information")
            st.write(f"**State:** {qubit}")
            st.write(f"**α (amplitude of |0⟩):** {qubit.alpha:.4f}")
            st.write(f"**β (amplitude of |1⟩):** {qubit.beta:.4f}")
            st.write(f"**P(0):** {qubit.prob_0():.4f}")
            st.write(f"**P(1):** {qubit.prob_1():.4f}")

            x, y, z = qubit.bloch_coordinates()
            st.write(f"**Bloch coordinates:** ({x:.3f}, {y:.3f}, {z:.3f})")

        with col2:
            st.markdown("### Bloch Sphere Visualization")
            fig = plot_bloch_matplotlib(qubit, "Your Custom Qubit")
            st.pyplot(fig)

    # ========================================================================
    # MODE 2: Gate Laboratory
    # ========================================================================
    elif mode == "🔧 Gate Laboratory":
        st.markdown('<h2 class="sub-header">Apply Quantum Gates</h2>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Select Initial State")
            initial_state = st.selectbox(
                "Choose starting state:",
                ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "Custom"]
            )

            if initial_state == "|0⟩":
                qubit = ket_0()
            elif initial_state == "|1⟩":
                qubit = ket_1()
            elif initial_state == "|+⟩":
                qubit = ket_plus()
            elif initial_state == "|−⟩":
                qubit = ket_minus()
            else:
                theta = st.slider("θ", 0.0, np.pi, np.pi/4, 0.01)
                phi = st.slider("φ", 0.0, 2*np.pi, 0.0, 0.01)
                qubit = create_custom_qubit(theta, phi)

            st.markdown("### Apply Gate")
            gate_choice = st.selectbox(
                "Select gate:",
                ["None", "X (NOT)", "Y", "Z", "H (Hadamard)", "S", "T",
                 "Rx(θ)", "Ry(θ)", "Rz(θ)"]
            )

            # Store original for comparison
            original_qubit = qubit.copy()

            # Apply gate
            if gate_choice != "None":
                if gate_choice == "X (NOT)":
                    qubit = apply_gate(qubit, PAULI_X)
                    gate_info = "Bit flip: |0⟩ ↔ |1⟩"
                elif gate_choice == "Y":
                    qubit = apply_gate(qubit, PAULI_Y)
                    gate_info = "Bit + phase flip"
                elif gate_choice == "Z":
                    qubit = apply_gate(qubit, PAULI_Z)
                    gate_info = "Phase flip: |1⟩ → -|1⟩"
                elif gate_choice == "H (Hadamard)":
                    qubit = apply_gate(qubit, HADAMARD)
                    gate_info = "Creates superposition"
                elif gate_choice == "S":
                    qubit = apply_gate(qubit, S_GATE)
                    gate_info = "90° phase shift"
                elif gate_choice == "T":
                    qubit = apply_gate(qubit, T_GATE)
                    gate_info = "45° phase shift"
                elif gate_choice.startswith("Rx"):
                    angle = st.slider("Rotation angle", 0.0, 2*np.pi, np.pi/2, 0.01)
                    qubit = apply_gate(qubit, rotation_x(angle))
                    gate_info = f"Rotation around X by {angle:.2f} rad"
                elif gate_choice.startswith("Ry"):
                    angle = st.slider("Rotation angle", 0.0, 2*np.pi, np.pi/2, 0.01)
                    qubit = apply_gate(qubit, rotation_y(angle))
                    gate_info = f"Rotation around Y by {angle:.2f} rad"
                elif gate_choice.startswith("Rz"):
                    angle = st.slider("Rotation angle", 0.0, 2*np.pi, np.pi/2, 0.01)
                    qubit = apply_gate(qubit, rotation_z(angle))
                    gate_info = f"Rotation around Z by {angle:.2f} rad"

                st.info(f"**Gate effect:** {gate_info}")

            st.markdown("### State Comparison")
            st.write(f"**Before:** {original_qubit}")
            st.write(f"**After:** {qubit}")

        with col2:
            st.markdown("### Bloch Sphere")

            # Create comparison plot
            bloch = BlochSphere(figsize=(8, 8))
            bloch.add_qubit(original_qubit, label="Before", color='blue')
            if gate_choice != "None":
                bloch.add_qubit(qubit, label="After", color='red')

            bloch._draw_sphere()
            bloch._draw_axes()
            bloch._draw_equator_and_meridians()
            bloch._draw_basis_states()
            bloch._plot_qubits()

            bloch.ax.set_xlabel('X')
            bloch.ax.set_ylabel('Y')
            bloch.ax.set_zlabel('Z')
            bloch.ax.set_title("Before vs After", fontsize=16, fontweight='bold')
            bloch.ax.set_box_aspect([1, 1, 1])
            bloch.ax.view_init(elev=20, azim=45)
            bloch.ax.set_xlim([-1.3, 1.3])
            bloch.ax.set_ylim([-1.3, 1.3])
            bloch.ax.set_zlim([-1.3, 1.3])
            bloch.ax.grid(False)

            st.pyplot(bloch.fig)

    # ========================================================================
    # MODE 3: Measurement Lab
    # ========================================================================
    elif mode == "📊 Measurement Lab":
        st.markdown('<h2 class="sub-header">Quantum Measurement Simulator</h2>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Create State to Measure")

            state_choice = st.selectbox(
                "Select state:",
                ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "Custom"]
            )

            if state_choice == "|0⟩":
                qubit = ket_0()
            elif state_choice == "|1⟩":
                qubit = ket_1()
            elif state_choice == "|+⟩":
                qubit = ket_plus()
            elif state_choice == "|−⟩":
                qubit = ket_minus()
            else:
                theta = st.slider("θ", 0.0, np.pi, np.pi/3, 0.01)
                phi = st.slider("φ", 0.0, 2*np.pi, 0.0, 0.01)
                qubit = create_custom_qubit(theta, phi)

            st.write(f"**State:** {qubit}")
            st.write(f"**P(0):** {qubit.prob_0():.4f}")
            st.write(f"**P(1):** {qubit.prob_1():.4f}")

            st.markdown("### Measurement Settings")
            shots = st.slider("Number of measurements:", 10, 10000, 1000, 10)

            if st.button("🎲 Measure!", type="primary"):
                st.session_state['measurement_done'] = True
                st.session_state['shots'] = shots
                st.session_state['qubit'] = qubit

        with col2:
            if st.session_state.get('measurement_done', False):
                st.markdown("### Measurement Results")
                qubit = st.session_state['qubit']
                shots = st.session_state['shots']

                fig = plot_measurement_histogram(qubit, shots)
                st.pyplot(fig)

                # Statistics
                outcomes = qubit.measure(shots=shots)
                count_0 = np.sum(outcomes == 0)
                count_1 = np.sum(outcomes == 1)

                st.markdown("### Statistics")
                st.write(f"**Total measurements:** {shots}")
                st.write(f"**Got |0⟩:** {count_0} times ({count_0/shots*100:.1f}%)")
                st.write(f"**Got |1⟩:** {count_1} times ({count_1/shots*100:.1f}%)")
                st.write(f"**Expected |0⟩:** {qubit.prob_0()*100:.1f}%")
                st.write(f"**Expected |1⟩:** {qubit.prob_1()*100:.1f}%")

    # ========================================================================
    # MODE 4: Common States
    # ========================================================================
    elif mode == "📚 Common States":
        st.markdown('<h2 class="sub-header">Important Quantum States</h2>',
                    unsafe_allow_html=True)

        states = {
            "|0⟩ (Computational basis)": ket_0(),
            "|1⟩ (Computational basis)": ket_1(),
            "|+⟩ (Hadamard basis)": ket_plus(),
            "|−⟩ (Hadamard basis)": ket_minus(),
            "|+i⟩ (Circular basis)": Qubit([1/np.sqrt(2), 1j/np.sqrt(2)]),
            "|−i⟩ (Circular basis)": Qubit([1/np.sqrt(2), -1j/np.sqrt(2)])
        }

        # Create multi-state visualization
        bloch = BlochSphere(figsize=(10, 10))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

        selected_states = st.multiselect(
            "Select states to visualize:",
            list(states.keys()),
            default=list(states.keys())[:3]
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            for i, state_name in enumerate(selected_states):
                qubit = states[state_name]
                bloch.add_qubit(qubit, label=state_name.split()[0],
                              color=colors[i % len(colors)])

                with st.expander(f"📋 {state_name}"):
                    st.write(f"**State:** {qubit}")
                    st.write(f"**α:** {qubit.alpha:.4f}")
                    st.write(f"**β:** {qubit.beta:.4f}")
                    x, y, z = qubit.bloch_coordinates()
                    st.write(f"**Bloch:** ({x:.3f}, {y:.3f}, {z:.3f})")

        with col2:
            bloch._draw_sphere()
            bloch._draw_axes()
            bloch._draw_equator_and_meridians()
            bloch._draw_basis_states()
            bloch._plot_qubits()

            bloch.ax.set_xlabel('X')
            bloch.ax.set_ylabel('Y')
            bloch.ax.set_zlabel('Z')
            bloch.ax.set_title("Common Quantum States", fontsize=16, fontweight='bold')
            bloch.ax.set_box_aspect([1, 1, 1])
            bloch.ax.view_init(elev=20, azim=45)
            bloch.ax.set_xlim([-1.3, 1.3])
            bloch.ax.set_ylim([-1.3, 1.3])
            bloch.ax.set_zlim([-1.3, 1.3])
            bloch.ax.grid(False)

            st.pyplot(bloch.fig)

    # ========================================================================
    # MODE 5: Gate Sequence
    # ========================================================================
    elif mode == "🎬 Gate Sequence":
        st.markdown('<h2 class="sub-header">Build a Gate Sequence</h2>',
                    unsafe_allow_html=True)

        st.markdown("""
        Build a sequence of gates and see the qubit trajectory on the Bloch sphere!
        """)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Initial State")
            initial = st.selectbox("Start with:", ["|0⟩", "|1⟩", "|+⟩", "|−⟩"])

            if initial == "|0⟩":
                qubit = ket_0()
            elif initial == "|1⟩":
                qubit = ket_1()
            elif initial == "|+⟩":
                qubit = ket_plus()
            else:
                qubit = ket_minus()

            st.markdown("### Gate Sequence")
            st.write("Select gates to apply in order:")

            gate_options = ["X", "Y", "Z", "H", "S", "T"]

            num_gates = st.slider("Number of gates:", 1, 10, 3)

            gates = []
            gate_names = []

            for i in range(num_gates):
                gate = st.selectbox(f"Gate {i+1}:", gate_options, key=f"gate_{i}")
                gate_names.append(gate)

                if gate == "X":
                    gates.append(PAULI_X)
                elif gate == "Y":
                    gates.append(PAULI_Y)
                elif gate == "Z":
                    gates.append(PAULI_Z)
                elif gate == "H":
                    gates.append(HADAMARD)
                elif gate == "S":
                    gates.append(S_GATE)
                elif gate == "T":
                    gates.append(T_GATE)

            st.write(f"**Circuit:** {initial} → " + " → ".join(gate_names))

            # Apply sequence
            final_qubit = apply_sequence(qubit, gates)

            st.markdown("### Result")
            st.write(f"**Initial:** {qubit}")
            st.write(f"**Final:** {final_qubit}")

        with col2:
            st.markdown("### Trajectory on Bloch Sphere")

            # Build trajectory
            bloch = BlochSphere(figsize=(9, 9))

            # Initial state
            bloch.add_qubit(qubit, label="Start", color='green')

            # Apply gates step by step
            current = qubit.copy()
            trajectory_points = [current.bloch_coordinates()]
            colors_seq = plt.cm.rainbow(np.linspace(0, 1, len(gates)))

            for i, (gate, name) in enumerate(zip(gates, gate_names)):
                current = apply_gate(current, gate)
                bloch.add_qubit(current, label=name, color=colors_seq[i])
                trajectory_points.append(current.bloch_coordinates())

            # Draw everything
            bloch._draw_sphere()
            bloch._draw_axes()
            bloch._draw_equator_and_meridians()
            bloch._plot_qubits()

            # Draw trajectory
            trajectory = np.array(trajectory_points)
            bloch.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                         'k--', linewidth=2, alpha=0.5)

            bloch.ax.set_xlabel('X')
            bloch.ax.set_ylabel('Y')
            bloch.ax.set_zlabel('Z')
            bloch.ax.set_title("Gate Sequence Trajectory", fontsize=16, fontweight='bold')
            bloch.ax.set_box_aspect([1, 1, 1])
            bloch.ax.view_init(elev=20, azim=45)
            bloch.ax.set_xlim([-1.3, 1.3])
            bloch.ax.set_ylim([-1.3, 1.3])
            bloch.ax.set_zlim([-1.3, 1.3])
            bloch.ax.grid(False)

            st.pyplot(bloch.fig)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>Built with Streamlit | Quantum Computing Learning Project</p>
    <p>Based on Imperial College London Quantum Information Notes</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
