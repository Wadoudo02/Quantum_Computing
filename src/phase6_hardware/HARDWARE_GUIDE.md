# Quantum Hardware Access Guide

Complete guide to accessing and using real quantum hardware from IBM, IonQ, Rigetti, and others.

---

## Table of Contents

1. [IBM Quantum](#ibm-quantum)
2. [IonQ (via AWS Braket)](#ionq-via-aws-braket)
3. [Rigetti (via Quantum Cloud Services)](#rigetti-via-quantum-cloud-services)
4. [Google Cirq](#google-cirq)
5. [Platform Comparison](#platform-comparison)
6. [Getting Started Checklist](#getting-started-checklist)
7. [Troubleshooting](#troubleshooting)

---

## IBM Quantum

### Overview
- **Technology**: Superconducting qubits
- **Access**: Free tier + paid plans
- **Best For**: Learning, research, algorithm development
- **Current Systems**: 127+ qubit processors

### Step 1: Create Account

1. Visit [https://quantum.ibm.com/](https://quantum.ibm.com/)
2. Click "Sign up" (can use IBM ID, Google, or GitHub)
3. Complete registration
4. Accept terms and conditions

### Step 2: Get API Token

1. Log in to IBM Quantum Experience
2. Click your profile icon (top right)
3. Go to "Account settings"
4. Find "API token" section
5. Click "Copy" to copy your token
6. **Important**: Keep this token secret!

### Step 3: Install Qiskit

```bash
# Install Qiskit
pip install qiskit qiskit-ibm-runtime

# Verify installation
python -c "import qiskit; print(qiskit.__version__)"
```

### Step 4: Configure Access

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your credentials (first time only)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_API_TOKEN_HERE",
    overwrite=True
)

# Load service
service = QiskitRuntimeService(channel="ibm_quantum")

# List available backends
backends = service.backends()
for backend in backends:
    print(f"{backend.name}: {backend.num_qubits} qubits")
```

### Step 5: Run Your First Circuit

```python
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# Create circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Get backend
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False)

# Transpile for hardware
qc_transpiled = transpile(qc, backend)

# Run using Sampler
sampler = Sampler(backend)
job = sampler.run(qc_transpiled, shots=1024)
result = job.result()

print(f"Counts: {result.quasi_dists[0]}")
```

### Free Tier Limits
- **10 minutes** of quantum processor time per month
- **Simulator**: Unlimited use
- **Queue priority**: Standard (may wait hours)
- **Jobs**: Up to 5 concurrent jobs

### Upgrade Options
- **IBM Quantum Premium**: More time, higher priority, dedicated support
- **Research partnerships**: For academic institutions
- **Commercial plans**: For enterprises

---

## IonQ (via AWS Braket)

### Overview
- **Technology**: Trapped ion qubits
- **Access**: Pay-per-shot via AWS
- **Best For**: High-fidelity requirements, full connectivity
- **Current Systems**: 32 qubits (Aria), 11 qubits (Harmony)

### Step 1: Create AWS Account

1. Visit [https://aws.amazon.com/](https://aws.amazon.com/)
2. Click "Create an AWS Account"
3. Complete registration
4. **Important**: Requires credit card (but has free tier)

### Step 2: Enable Amazon Braket

1. Log in to AWS Console
2. Search for "Amazon Braket"
3. Click "Get started"
4. Accept service terms
5. Enable Braket in your preferred region (us-east-1 recommended)

### Step 3: Set Up AWS CLI

```bash
# Install AWS CLI
pip install awscli boto3

# Configure credentials
aws configure
# Enter:
#   AWS Access Key ID: [from IAM]
#   AWS Secret Access Key: [from IAM]
#   Default region: us-east-1
#   Default output format: json
```

### Step 4: Install Amazon Braket SDK

```bash
# Install Braket SDK
pip install amazon-braket-sdk

# Verify installation
python -c "import braket; print(braket.__version__)"
```

### Step 5: Run Circuit on IonQ

```python
from braket.aws import AwsDevice
from braket.circuits import Circuit

# Create circuit
circuit = Circuit().h(0).cnot(0, 1)

# Select IonQ device
device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")

# Run (this costs money!)
task = device.run(circuit, shots=100)

# Wait for result
result = task.result()
print(f"Counts: {result.measurement_counts}")
```

### Pricing (as of 2024)
- **IonQ Harmony**: $0.30 per task + $0.01 per shot
- **IonQ Aria**: $0.30 per task + $0.03 per shot
- **Simulators**: Free (within free tier limits)

**Example Cost Calculation:**
- 100 shots on Harmony: $0.30 + ($0.01 × 100) = $1.30
- 1000 shots on Aria: $0.30 + ($0.03 × 1000) = $30.30

### Cost Management Tips
1. **Use simulators first**: Test on free simulators before hardware
2. **Start small**: Begin with 100 shots to verify correctness
3. **Set budgets**: Use AWS Budgets to track spending
4. **Use local simulator**: Test locally before submitting to AWS

---

## Rigetti (via Quantum Cloud Services)

### Overview
- **Technology**: Superconducting qubits
- **Access**: QCS (Quantum Cloud Services)
- **Best For**: Low-latency applications, hybrid algorithms
- **Current Systems**: 80+ qubit Aspen series

### Step 1: Request Access

1. Visit [https://qcs.rigetti.com/](https://qcs.rigetti.com/)
2. Click "Request Access"
3. Fill out application form
   - Academic users: Usually approved quickly
   - Commercial: May require business justification
4. Wait for approval email (typically 1-3 business days)

### Step 2: Set Up pyQuil

```bash
# Install pyQuil (Rigetti's SDK)
pip install pyquil

# Install Quil compiler
pip install quantum-quil-compiler
```

### Step 3: Configure QCS

```bash
# Download QCS credentials
# Instructions provided in approval email

# Set up configuration
qcs setup
# Follow prompts to enter:
#   - API token
#   - User ID
#   - Preferred compiler URL
```

### Step 4: Run Circuit

```python
from pyquil import Program, get_qc
from pyquil.gates import H, CNOT, MEASURE

# Create program
p = Program()
p += H(0)
p += CNOT(0, 1)
ro = p.declare('ro', 'BIT', 2)
p += MEASURE(0, ro[0])
p += MEASURE(1, ro[1])

# Get quantum computer
qc = get_qc('Aspen-M-3')  # or use '2q-qvm' for simulator

# Compile and run
compiled = qc.compile(p)
result = qc.run(compiled)

print(f"Results: {result}")
```

### Access Levels
- **Free tier**: Limited simulator access
- **Academic**: Discounted or free hardware access
- **Commercial**: Full access with various plans

---

## Google Cirq

### Overview
- **Technology**: Superconducting qubits (Sycamore)
- **Access**: Limited (research partnerships only)
- **Best For**: Research, Google collaborations
- **Current Systems**: 54 qubits (Sycamore)

### Important Note
Google's quantum hardware is **not publicly available** as of 2024. Access is granted only to:
- Research partners
- Academic collaborations
- Select commercial partnerships

### Using Cirq (Simulation Only)

```bash
# Install Cirq
pip install cirq

# Verify installation
python -c "import cirq; print(cirq.__version__)"
```

```python
import cirq

# Create circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)

# Simulate (no hardware access)
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)

print(f"Counts: {result.histogram(key='result')}")
```

---

## Platform Comparison

### Hardware Specifications

| Platform | Technology | Qubits | T₁ (μs) | T₂ (μs) | 1Q Fidelity | 2Q Fidelity | Connectivity |
|----------|------------|--------|---------|---------|-------------|-------------|--------------|
| **IBM** | Superconducting | 127 | ~100-150 | ~80-120 | 99.95% | 98.7% | Limited (line/hex) |
| **IonQ** | Trapped Ions | 32 | ~1,000,000 | ~500,000 | 99.98% | 97.2% | All-to-all |
| **Rigetti** | Superconducting | 80 | ~15-30 | ~10-25 | 99.8% | 90% | Limited (line) |
| **Google** | Superconducting | 54 | ~20-40 | ~15-30 | 99.9% | 94% | Grid |

### Access & Cost

| Platform | Public Access | Cost Model | Free Tier | Best For |
|----------|---------------|------------|-----------|----------|
| **IBM** | ✅ Yes | Free + Premium | 10 min/month | Learning, research |
| **IonQ** | ✅ Yes (via AWS) | Pay-per-shot | Simulators only | High-fidelity apps |
| **Rigetti** | ✅ Yes (request) | Subscription | Limited simulator | Hybrid algorithms |
| **Google** | ❌ No | Partnership only | N/A | Research only |

### SDK Features

| Feature | IBM (Qiskit) | IonQ (Braket) | Rigetti (pyQuil) | Google (Cirq) |
|---------|--------------|---------------|------------------|---------------|
| **Language** | Python | Python | Python | Python |
| **Simulators** | ✅ Excellent | ✅ Good | ✅ Good | ✅ Excellent |
| **Visualization** | ✅ Excellent | ⚠️ Basic | ⚠️ Basic | ✅ Good |
| **Optimization** | ✅ Extensive | ⚠️ Limited | ✅ Good | ✅ Good |
| **Documentation** | ✅ Excellent | ✅ Good | ⚠️ Limited | ✅ Excellent |
| **Community** | ✅ Large | ⚠️ Growing | ⚠️ Small | ✅ Large |

### Which Platform to Choose?

**For Learning & Experimentation:**
- → **IBM Quantum** (free tier, great documentation, large community)

**For High-Fidelity Requirements:**
- → **IonQ** (best gate fidelities, all-to-all connectivity)

**For Hybrid Algorithms (VQE, QAOA):**
- → **Rigetti** (optimized for hybrid quantum-classical)

**For Maximum Qubit Count:**
- → **IBM Quantum** (127 qubits currently)

**For Research:**
- → **IBM or Google** (best academic support)

---

## Getting Started Checklist

### Before You Begin

- [ ] Choose your primary platform (IBM recommended for beginners)
- [ ] Create account and get API credentials
- [ ] Install required SDKs (`pip install qiskit` or equivalent)
- [ ] Verify installation with simple example
- [ ] Test on simulator first

### First Steps

- [ ] Run "Hello World" Bell state circuit
- [ ] Check queue times and availability
- [ ] Submit first hardware job (if using IBM's free tier)
- [ ] Compare simulator vs hardware results
- [ ] Understand noise and errors in your results

### Best Practices

- [ ] Always test on simulator first
- [ ] Start with small circuits (< 10 gates)
- [ ] Use error mitigation when available
- [ ] Monitor queue times (run jobs during off-peak hours)
- [ ] Save and version control your circuits
- [ ] Document hardware parameters (date, backend, noise characteristics)

---

## Troubleshooting

### IBM Quantum

**Problem**: "Job failed with status: CANCELLED"
- **Solution**: Backend may have gone offline. Check system status at [https://quantum-computing.ibm.com/services/resources](https://quantum-computing.ibm.com/services/resources)

**Problem**: Long queue times (hours)
- **Solution**: Use `service.least_busy()` to find backend with shortest queue

**Problem**: "Invalid API token"
- **Solution**: Regenerate token and save again with `QiskitRuntimeService.save_account()`

### IonQ (AWS Braket)

**Problem**: "InsufficientPermissionsException"
- **Solution**: Ensure your IAM role has AmazonBraketFullAccess policy

**Problem**: High costs
- **Solution**: Use simulators for development. Only run hardware jobs when needed.

**Problem**: "DeviceOfflineException"
- **Solution**: Check device availability at [AWS Braket console](https://console.aws.amazon.com/braket)

### Rigetti

**Problem**: "QCS authorization failed"
- **Solution**: Run `qcs setup` again and re-enter credentials

**Problem**: "Compiler service unavailable"
- **Solution**: Check QCS status page or try again later

### General

**Problem**: Results don't match theory
- **Solution**: This is normal! Real hardware has noise. Use error mitigation.

**Problem**: Circuit won't compile
- **Solution**: Check connectivity constraints. Use transpiler to insert SWAPs.

**Problem**: Out of credits/quota
- **Solution**: Wait for quota refresh or upgrade to paid tier

---

## Additional Resources

### Official Documentation
- **IBM Qiskit**: [https://qiskit.org/documentation/](https://qiskit.org/documentation/)
- **AWS Braket**: [https://docs.aws.amazon.com/braket/](https://docs.aws.amazon.com/braket/)
- **Rigetti QCS**: [https://docs.rigetti.com/](https://docs.rigetti.com/)
- **Google Cirq**: [https://quantumai.google/cirq](https://quantumai.google/cirq)

### Community Forums
- **IBM Quantum**: [Qiskit Slack](https://qiskit.slack.com/)
- **AWS Braket**: [AWS re:Post](https://repost.aws/)
- **Rigetti**: [Rigetti Community Forum](https://community.rigetti.com/)
- **General**: [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/)

### Tutorials & Examples
- **Qiskit Textbook**: [https://qiskit.org/textbook/](https://qiskit.org/textbook/)
- **AWS Braket Examples**: [GitHub repo](https://github.com/aws/amazon-braket-examples)
- **Rigetti Tutorials**: [QCS Documentation](https://docs.rigetti.com/qcs/getting-started/using-quil)

---

## Security Best Practices

1. **Never commit API tokens to Git**
   ```bash
   # Add to .gitignore
   echo "*token*" >> .gitignore
   echo "*.env" >> .gitignore
   ```

2. **Use environment variables**
   ```python
   import os
   token = os.getenv('IBM_QUANTUM_TOKEN')
   ```

3. **Rotate credentials regularly**
   - Regenerate API tokens every 3-6 months
   - Revoke old tokens immediately

4. **Monitor usage**
   - Check for unexpected jobs
   - Set up billing alerts (for paid services)

---

## Summary

You now have all the information needed to access real quantum hardware! Start with:

1. **IBM Quantum** (free, easy to set up)
2. Run simulators first
3. Submit small test jobs to hardware
4. Analyze results and compare with theory
5. Explore error mitigation techniques

**Remember**: Real quantum computers are noisy. Expect results to differ from theory—that's what makes NISQ computing interesting!

---

*Last Updated: November 2024*
*For: Quantum Computing Learning Project - Phase 6*
