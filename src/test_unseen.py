"""
Unseen Resume Testing — P3.10
Author: Savan (Person 3)
Project: Resume Classification Hackathon (SAMATRIX RESUMEFORGE 2026)

Tests the final pipeline on 5 unseen resume examples that were NOT in the dataset.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows consoles default to cp1252 and cannot render the box-drawing/emoji glyphs
# used in the report headers; force UTF-8 so the script runs anywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.predict import ResumeClassifierPipeline


UNSEEN_RESUMES = [
    {
        "expected": "INFORMATION-TECHNOLOGY",
        "text": """
        Senior Software Engineer with 8+ years of experience in full-stack development.
        Proficient in Python, Java, JavaScript, React, Node.js, and AWS cloud services.
        Led a team of 12 engineers building scalable microservices architecture using Docker,
        Kubernetes, and CI/CD pipelines. Implemented real-time data processing systems
        handling 10M+ daily transactions. Experience with PostgreSQL, MongoDB, Redis.
        Bachelor of Science in Computer Science from MIT. Certified AWS Solutions Architect.
        """,
    },
    {
        "expected": "ACCOUNTANT",
        "text": """
        Certified Public Accountant (CPA) with 6 years of experience in financial auditing,
        tax preparation, and general ledger management. Expertise in GAAP compliance,
        financial reporting, and internal controls. Managed audits for Fortune 500 clients
        with combined revenue exceeding $2B. Proficient in QuickBooks, SAP, and Excel.
        Master of Accounting from University of Texas at Austin.
        """,
    },
    {
        "expected": "CHEF",
        "text": """
        Executive Chef with 12 years of culinary experience in fine dining restaurants.
        Expertise in French, Italian, and Asian fusion cuisines. Managed kitchen operations
        for a Michelin-starred restaurant with 45 staff members. Specialized in pastry arts,
        molecular gastronomy, and farm-to-table menu development. Culinary degree from
        Le Cordon Bleu. ServSafe certified. Experience in inventory control, vendor management,
        and menu cost optimization achieving 28% food cost ratio.
        """,
    },
    {
        "expected": "HEALTHCARE",
        "text": """
        Registered Nurse (RN) with 5 years of experience in emergency medicine and critical care.
        BSN from Johns Hopkins University School of Nursing. ACLS, BLS, and PALS certified.
        Managed patient assessments, IV therapy, medication administration, and wound care
        for 15+ patients per shift. Experience with Epic EMR system. Specialized in trauma
        nursing and cardiac monitoring. CPR instructor certification.
        """,
    },
    {
        "expected": "ENGINEERING",
        "text": """
        Mechanical Engineer with 7 years of experience in product design and manufacturing.
        Proficient in AutoCAD, SolidWorks, CATIA, and ANSYS for FEA simulation.
        Designed automotive components reducing weight by 15% while maintaining structural
        integrity. Experience in GD&T, tolerance analysis, and DFM/DFA principles.
        Six Sigma Green Belt certified. Master of Engineering from Stanford University.
        Led cross-functional teams of 8 engineers in new product development programs.
        """,
    },
]


def run_unseen_tests():
    print("=" * 70)
    print("  P3.10: Testing on 5 Unseen Resume Examples")
    print("=" * 70)

    pipeline = ResumeClassifierPipeline()

    correct = 0
    total = len(UNSEEN_RESUMES)

    for i, sample in enumerate(UNSEEN_RESUMES, 1):
        result = pipeline.predict(sample["text"])
        predicted = result["predicted_category"]
        expected = sample["expected"]
        match = "[PASS]" if predicted == expected else "[FAIL]"

        if predicted == expected:
            correct += 1

        print(f"\n--- Resume {i}/{total} ---")
        print(f"  Expected:  {expected}")
        print(f"  Predicted: {predicted} {match}")
        print(f"  Confidence: {result.get('confidence', 0) * 100:.1f}%")
        print(f"  Pipeline: {result.get('pipeline_type', 'N/A')}")

    print(f"\n{'=' * 70}")
    print(f"  UNSEEN TEST RESULTS: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print(f"{'=' * 70}")

    return correct, total


if __name__ == "__main__":
    run_unseen_tests()
