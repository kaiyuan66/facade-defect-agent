Use the following CoT process internally and only output the final structured result:

Step 1: Parse multimodal inputs: RGB image, thermal image, bbox geometry, and X/Y/Z/Z_alt/Area metadata.
Step 2: Decide Severity first (MINOR / MODERATE / MAJOR only). Then choose DefectType (no Unknown; use Unspecified defect if needed).
Step 3: Write Task A analysis sections using multimodal evidence.
Step 4: Do not use LOW, HIGH, CRITICAL, or UNKNOWN as severity labels.

Output template (exact order):
Severity: <MINOR|MODERATE|MAJOR>
DefectType: <label>
- Defect Identification:
- Risk Level:
- Cause Analysis:
- Maintenance Strategy:
- Uncertainty and Follow-up Actions:
