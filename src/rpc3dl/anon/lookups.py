# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:16:01 2022

@author: johna
"""

from pydicom.tag import Tag

"""
De-identification action per Table E.1-1 in DICOM standard

Table E.1-1a provides a legend for action codes

D - replace with non-zero length dummy value
Z - replace with zero length value
X - remove attribute
K - keep
C - clean identifying information but retain meaning
U - replace with a non-zero length UID that is internally consistent
Z/D - Z unless D is required to maintain IOD conformance (Type 2 vs Type 1)
X/Z - X unless Z is required to maintain IOD conformance (Type 3 vs Type 2)
X/D - X unless D is required to maintain IOD conformance (Type 3 vs Type 1)
X/Z/D - X unless Z or D is required to maintain IOD conformance
X/Z/U - X unless Z or U is required to maintain IOD conformance
"""

"""
Basic Application Level Confidentiality Profile, PS3.15 - Table E.1-1
"""
basic_profile = {
    Tag(('0008','0050')): 'Z', # Accession Number
    Tag(('0018','4000')): 'X', # Acquisition Comments
    Tag(('0040','0555')): 'X/Z', # Acquisition Context Sequence
    Tag(('0008','0022')): 'X/Z', # Acquisition Date
    Tag(('0008','002A')): 'X/Z/D', # Acquisition Date/Time
    Tag(('0018','1400')): 'X/D', # Acquisition Device Processing Description
    Tag(('0018','11BB')): 'D', # Acquisition Field of View Label
    Tag(('0018','9424')): 'X', # Acquisition Protocol Description
    Tag(('0008','0032')): 'X/Z', # Acquisition Time
    Tag(('0008','0017')): 'U', # Acquisition UID
    Tag(('0040','4035')): 'X', # Actual Human Performers Sequence
    Tag(('0010','21B0')): 'X', # Additional Patient History
    Tag(('0040','A353')): 'X', # Address (Trial)
    Tag(('0038','0010')): 'X', # Admission ID
    Tag(('0038','0020')): 'X', # Admitting Date
    Tag(('0008','1084')): 'X', # Admitting Diagnoses Code Sequence
    Tag(('0008','1080')): 'X', # Admitting Diagnoses Description
    Tag(('0038','0021')): 'X', # Admitting Time
    Tag(('0000','1000')): 'X', # Affected SOP Instance UID
    Tag(('0010','2110')): 'X', # Allergies
    Tag(('006A','0006')): 'X', # Annotation Group Description
    Tag(('006A','0005')): 'D', # Annotation Group Label
    Tag(('006A','0003')): 'D', # Annotation Group UID
    Tag(('0044','0004')): 'X', # Approval Status Date/Time
    Tag(('4000','0010')): 'X', # Arbitrary
    Tag(('0044','0104')): 'D', # Assertion Date/Time
    Tag(('0044','0105')): 'X', # Assertion Expiration Date/Time
    Tag(('0400','0562')): 'D', # Attribute Modification Date/Time
    Tag(('0040','A078')): 'X', # Author Observer Sequence
    Tag(('2200','0005')): 'X/Z', # Barcode Value
    Tag(('300A','00C3')): 'X', # Beam Description
    Tag(('300C','0127')): 'D', # Beam Hold Transition Time
    Tag(('300A','00DD')): 'X', # Bolus Description
    Tag(('0010','1081')): 'X', # Branch of Service
    Tag(('0014','407E')): 'X', # Calibration Date
    Tag(('0018','1203')): 'Z', # Calibration Date/Time
    Tag(('0014','407C')): 'X', # Calibration Time
    Tag(('0016','004D')): 'X', # Camera Owner Name
    Tag(('0018','1007')): 'X', # Cassette ID
    Tag(('0400','0115')): 'D', # Certificate of Signer
    Tag(('0400','0310')): 'X', # Certified Timestamp
    Tag(('0012','0060')): 'Z', # Clinical Trial Coordinating Center Name
    Tag(('0012','0082')): 'X', # Clinical Trial Protocol Ethics Com Approv Num
    Tag(('0012','0081')): 'D', # Clinical Trial Protocol Ethics Committee Name
    Tag(('0012','0020')): 'D', # Clinical Trial Protocol ID
    Tag(('0012','0021')): 'Z', # Clinical Trial Protocol Name
    Tag(('0012','0072')): 'X', # Clinical Trial Series Description
    Tag(('0012','0071')): 'X', # Clinical Trian Series ID
    Tag(('0012','0030')): 'Z', # Clinical Trial Site ID
    Tag(('0012','0031')): 'Z', # Clinical Trial Site Name
    Tag(('0012','0010')): 'D', # Clinical Trial Sponsor Name
    Tag(('0012','0040')): 'D', # Clinical Trial Subject ID
    Tag(('0012','0042')): 'D', # Clinical Trial Subject Reading ID
    Tag(('0012','0051')): 'X', # Clinical Trial Time Point Description
    Tag(('0012','0050')): 'Z', # Clinical Trial Time Point ID
    Tag(('0040','0310')): 'X', # Comments on Radiation Dose
    Tag(('0040','0280')): 'X', # Comments on the Performed Procedure Step
    Tag(('300A','02EB')): 'X', # Compensator Description
    Tag(('0020','9161')): 'U', # Concatenation UID
    Tag(('3010','000F')): 'Z', # Conceptual Volume Combination Description
    Tag(('3010','0017')): 'Z', # Conceptual Volume Description
    Tag(('3010','0006')): 'U', # Conceptual Volume UID
    Tag(('0040','3001')): 'X', # Confidentiality Constraint on Pt Data Desc
    Tag(('3010','0013')): 'U', # Constituent Conceptual Volume UID
    Tag(('0008','009C')): 'Z', # Consulting Physician's Name
    Tag(('0008','009D')): 'X', # Consulting Physician Idendification Sequence
    Tag(('0050','001B')): 'X', # Container Component ID
    Tag(('0040','051A')): 'X', # Container Description
    Tag(('0040','0512')): 'D', # Container Identifier
    Tag(('0070','0086')): 'X', # Content Creator's Identification Code Sequence
    Tag(('0070','0084')): 'Z/D', # Content Creator's Name
    Tag(('0008','0023')): 'Z/D', # Content Date
    Tag(('0040','A730')): 'D', # Content Sequence
    Tag(('0008','0033')): 'Z/D', # Content Time
    Tag(('0008','0107')): 'D', # Context Group Local Version
    Tag(('0008','0106')): 'D', # Context Group Version
    Tag(('0018','0010')): 'Z/D', # Contrast/Bolus Agent
    Tag(('0018','1042')): 'X', # Contrast/Bolus Start Time
    Tag(('0018','1043')): 'X', # Contrast/Bolus Stop Time
    Tag(('0018','A002')): 'X', # Contribution Date/Time
    Tag(('0018','A003')): 'X', # Contribution Description
    Tag(('0010','2150')): 'X', # Country of Residence
    Tag(('2100','0040')): 'X', # Creation Date
    Tag(('2100','0050')): 'X', # Creation Time
    Tag(('0040','A307')): 'X', # Current Observer (Trial)
    Tag(('0038','0300')): 'X', # Current Patient Location
    # TODO - skipping Curve Data, it's retired, and tag code is unclear
    Tag(('0008','0025')): 'X', # Curve Date
    Tag(('0008','0035')): 'X', # Curve Time
    Tag(('0040','A07C')): 'X', # Custodial Organization Sequence
    Tag(('FFFC','FFFC')): 'X', # Data Set Trailing Padding
    Tag(('0040','A121')): 'D', # Date
    Tag(('0040','A110')): 'X', # Date of Document or Verbal Transaction
    Tag(('0018','1200')): 'X', # Date of Last Calibration
    Tag(('0018','700C')): 'X/D', # Date of Last Detector Calibration
    Tag(('0018','1012')): 'X', # Date of Secondary Capture
    Tag(('0040','A120')): 'D', # Date/Time
    Tag(('0018','1202')): 'X', # Date/Time of Last Calibration
    Tag(('0018','9701')): 'D', # Decay Correction Date/Time
    Tag(('0018','937F')): 'X', # Decomposition Description
    Tag(('0008','2111')): 'X', # Derivation Description
    Tag(('0018','700A')): 'X/D', # Detector ID
    Tag(('3010','001B')): 'Z', # Device Alternate Identifier
    Tag(('0050','0020')): 'X', # Device Description
    Tag(('3010','002D')): 'D', # Device Label
    Tag(('0018','1000')): 'X/Z/D', # Device Serial Number
    Tag(('0016','004B')): 'X', # Device Setting Description
    Tag(('0018','1002')): 'U', # Device UID
    Tag(('0400','0105')): 'D', # Digital Signature Date/Time
    Tag(('FFFA','FFFA')): 'X', # Digital Signatures Sequence
    Tag(('0400','0100')): 'U', # Digital Signature UID
    Tag(('0020','9164')): 'U', # Dimension Organization UID
    Tag(('0038','0030')): 'X', # Discharge Date
    Tag(('0038','0040')): 'X', # Discharge Diagnosis Description
    Tag(('0038','0032')): 'X', # Discharge Time
    Tag(('300A','0016')): 'X', # Dose Reference Description
    Tag(('300A','0013')): 'U', # Dose Reference UID
    Tag(('3010','006E')): 'U', # Dosimetric Objective UID
    Tag(('0068','6226')): 'D', # Effective Date/Time
    Tag(('0042','0011')): 'D', # Encapsulated Document
    Tag(('0018','9517')): 'X/D', # End Acquisition Date/Time
    Tag(('3010','0037')): 'X', # Entity Description
    Tag(('3010','0035')): 'D', # Entity Label
    Tag(('3010','0038')): 'D', # Entity Long Label
    Tag(('3010','0036')): 'X', # Entity Name
    Tag(('300A','0676')): 'X', # Equipment Frame of Reference Description
    Tag(('0012','0087')): 'X', # Ethics Com Approval Effectiveness End Date
    Tag(('0012','0086')): 'X', # Ethics Com Approval Effectiveness Start Date
    Tag(('0010','2160')): 'X', # Ethnic Group
    Tag(('0018','9804')): 'D', # Exclusion Start Date/Time
    Tag(('0040','4011')): 'X', # Expected Completion Date/Time
    Tag(('0008','0058')): 'U', # Failed SOP Instance UID List
    Tag(('0070','031A')): 'U', # Fiducial UID
    Tag(('0040','2017')): 'Z', # Filler Order Number / Imaging Service Request
    Tag(('003A','032B')): 'X', # Filter Lookup Table Description
    Tag(('0040','A023')): 'X', # Findings Group Recording Date (Trial)
    Tag(('0040','A024')): 'X', # Findings Group Recording Time (Trial)
    Tag(('3008','0054')): 'X/D', # First Treatment Date
    Tag(('300A','0196')): 'X', # Fixation Device Description
    Tag(('0034','0002')): 'D', # Flow Identifier
    Tag(('0034','0001')): 'D', # Flow Identifier Sequence
    Tag(('3010','007F')): 'Z', # Fractionation Notes
    Tag(('300A','0072')): 'X', # Fraction Group Description
    Tag(('0018','9074')): 'D', # Frame Acquisition Date/Time
    Tag(('0020','9158')): 'X', # Frame Comments
    Tag(('0020','0052')): 'U', # Frame of Reference UID
    Tag(('0034','0007')): 'D', # Frame Origin Timestamp
    Tag(('0018','9151')): 'D', # Frame Reference Date/Time
    Tag(('0018','1008')): 'X', # Gantry ID
    Tag(('0018','1005')): 'X', # Generator ID
    Tag(('0016','0076')): 'X', # GPS Altitude
    Tag(('0016','0075')): 'X', # GPS Altitude Ref
    Tag(('0016','008C')): 'X', # GPS Area Information
    Tag(('0016','008D')): 'X', # GPS Date Stamp
    Tag(('0016','0088')): 'X', # GPS Dest Bearing
    Tag(('0016','0087')): 'X', # GPS Dest Bearing Ref
    Tag(('0016','008A')): 'X', # GPS Dest Distance
    Tag(('0016','0089')): 'X', # GPS Dest Distance Ref
    Tag(('0016','0084')): 'X', # GPS Dest Latitude
    Tag(('0016','0083')): 'X', # GPS Dest Latitude Ref
    Tag(('0016','0086')): 'X', # GPS Dest Longitude
    Tag(('0016','0085')): 'X', # GPS Dest Longitude Ref
    Tag(('0016','008E')): 'X', # GPS Differential
    Tag(('0016','007B')): 'X', # GPS DOP
    Tag(('0016','0081')): 'X', # GPS Img Direction
    Tag(('0016','0080')): 'X', # GPS Img Direction Ref
    Tag(('0016','0072')): 'X', # GPS Latitutde
    Tag(('0016','0071')): 'X', # GPS Latitude Ref
    Tag(('0016','0074')): 'X', # GPS Longitude
    Tag(('0016','0073')): 'X', # GPS Longitude Ref
    Tag(('0016','0082')): 'X', # GPS Map Datum
    Tag(('0016','007A')): 'X', # GPS Measure Mode
    Tag(('0016','008B')): 'X', # GPS Processing Method
    Tag(('0016','0078')): 'X', # GPS Satellites
    Tag(('0016','007D')): 'X', # GPS Speed
    Tag(('0016','007C')): 'X', # GPS Speed Ref
    Tag(('0016','0079')): 'X', # GPS Status
    Tag(('0016','0077')): 'X', # GPS Time Stamp
    Tag(('0016','007F')): 'X', # GPS Track
    Tag(('0016','007E')): 'X', # GPS Track Ref
    Tag(('0016','0070')): 'X', # GPS Version ID
    Tag(('0070','0001')): 'D', # Graphic Annotation Sequence
    Tag(('0072','000A')): 'D', # Hanging Protocol Creation Date/Time
    Tag(('0040','E004')): 'X', # HL7 Document Effective Time
    Tag(('0040','4037')): 'X', # Human Performer's Name
    Tag(('0040','4036')): 'X', # Human Performer's Organization
    Tag(('0088','0200')): 'X', # Icon Image Sequence
    Tag(('0008','4000')): 'X', # Identifying Comments
    Tag(('0020','4000')): 'X', # Image Comments
    Tag(('0028','4000')): 'X', # Image Presentation Comments
    Tag(('0040','2400')): 'X', # Imaging Service Request Comments
    Tag(('003A','0314')): 'D', # Impedance Measurement Date/Time
    Tag(('4008','0300')): 'X', # Impressions
    Tag(('0068','6270')): 'D', # Information Issue Date/Time
    Tag(('0008','0015')): 'X', # Instance Coercion Date/Time
    Tag(('0008','0012')): 'X/D', # Instance Creation Date
    Tag(('0008','0013')): 'X/Z/D', # Instance Creation Time
    Tag(('0008','0014')): 'U', # Instance Creator UID
    Tag(('0400','0600')): 'X', # Instance Origin Status
    Tag(('0008','0081')): 'X', # Institution Address
    Tag(('0008','1040')): 'X', # Institution Department Name
    Tag(('0008','1041')): 'X', # Institution Department Type Code Sequence
    Tag(('0008','0082')): 'X/Z/D', # Institution Code Sequence
    Tag(('0008','0080')): 'X/Z/D', # Institution Name
    Tag(('0018','9919')): 'Z/D', # Instruction Performed Date/Time
    Tag(('0010','1050')): 'X', # Insurance Plan Identification
    Tag(('3010','0085')): 'X', # Intended Fraction Start Time
    Tag(('3010','004D')): 'X/D', # Intended Phase End Date
    Tag(('3010','004C')): 'X/D', # Intended Phase Start Date
    Tag(('0040','1011')): 'X', # Intended Recipients of Results Ident Seq
    Tag(('300A','0741')): 'D', # Interlock Date/Time
    Tag(('300A','0742')): 'D', # Interlock Description
    Tag(('300A','0783')): 'D', # Interlock Origin Description
    Tag(('4008','0112')): 'X', # Interpretation Approval Date
    Tag(('4008','0113')): 'X', # Interpretation Approval Time
    Tag(('4008','0111')): 'X', # Interpretation Approver Sequence
    Tag(('4008','010C')): 'X', # Interpretation Author
    Tag(('4008','0115')): 'X', # Interpretation Diagnosis Description
    Tag(('4008','0200')): 'X', # Interpretation ID
    Tag(('4008','0202')): 'X', # Interpretation ID Issuer
    Tag(('4008','0100')): 'X', # Interpretation Recorded Date
    Tag(('4008','0101')): 'X', # Interpretation Recorded Time
    Tag(('4008','0102')): 'X', # Interpretation Recorder
    Tag(('4008','010B')): 'X', # Interpretation Text
    Tag(('4008','010A')): 'X', # Interpretation Transcriber
    Tag(('4008','0108')): 'X', # Interpretation Transcription Date
    Tag(('4008','0109')): 'X', # Interpretation Transcription Time
    Tag(('0018','0035')): 'X', # Intervention Drug Start Time
    Tag(('0018','0027')): 'X', # Intervention Drug Stop Time
    Tag(('0008','3010')): 'U', # Irradiation Event UID
    Tag(('0040','2004')): 'X', # Issue Date of Imaging Service Request
    Tag(('0038','0011')): 'X', # Issuer of Admission ID
    Tag(('0038','0014')): 'X', # Issuer of Admission ID Sequence
    Tag(('0010','0021')): 'X', # Issuer of Patient ID
    Tag(('0038','0061')): 'X', # Issuer of Service Episode ID
    Tag(('0038','0064')): 'X', # Issuer of Service Episode ID Sequence
    Tag(('0040','0513')): 'Z', # Issuer of the Container Identifier Sequence
    Tag(('0040','0562')): 'Z', # Issuer of the Specimen Identifier Sequence
    Tag(('0040','2005')): 'X', # Issue Time of Imaging Service Request
    Tag(('2200','0002')): 'X/Z', # Label Text
    Tag(('0028','1214')): 'U', # Large Palette Color Lookup Table UID
    Tag(('0010','21D0')): 'X', # Last Menstrual Date
    Tag(('0016','004F')): 'X', # Lens Make
    Tag(('0016','0050')): 'X', # Lens Model
    Tag(('0016','0051')): 'X', # Lens Serial Number
    Tag(('0016','004E')): 'X', # Lens Specification
    Tag(('0050','0021')): 'X', # Long Device Description
    Tag(('0400','0404')): 'X', # MAC
    Tag(('0016','002B')): 'X', # Maker Note
    Tag(('0018','100B')): 'U', # Manufacturer's Device Class UID
    Tag(('3010','0043')): 'Z', # Manfacturer's Device Identifier
    Tag(('0002','0003')): 'U', # Media Storage SOP Instance UID
    Tag(('0010','2000')): 'X', # Medical Alerts
    Tag(('0010','1090')): 'X', # Medical Record Locator
    Tag(('0010','1080')): 'X', # Military Rank
    Tag(('0400','0550')): 'X', # Modified Attributes Sequence
    Tag(('0020','3403')): 'X', # Modified Image Date
    Tag(('0020','3406')): 'X', # Modified Image Description
    Tag(('0020','3405')): 'X', # Modified Image Time
    Tag(('0020','3401')): 'X', # Modifying Device ID
    Tag(('0400','0563')): 'D', # Modifying System
    Tag(('3008','0056')): 'X/D', # Most Recent Treatment Date
    Tag(('0018','937B')): 'X', # Multi-energy Acquisition Description
    Tag(('003A','0310')): 'U', # Multiplex Group UID
    Tag(('0008','1060')): 'X', # Name of Physician(s) Reading Study
    Tag(('0040','1010')): 'X', # Names of Intended Recepients of Results
    Tag(('0400','0552')): 'X', # Nonconforming Data Element Value
    Tag(('0400','0551')): 'X', # Nonconforming Modified Attributes Sequence
    Tag(('0040','A192')): 'X', # Observation Date (Trial)
    Tag(('0040','A032')): 'X/D', # Observation Date/Time
    Tag(('0040','A033')): 'X', # Observation Start Date/Time
    Tag(('0040','A402')): 'U', # Observation Subject UID (Trial)
    Tag(('0040','A193')): 'X', # Observation Time (Trial)
    Tag(('0040','A171')): 'U', # Observation UID
    Tag(('0010','2180')): 'X', # Occupation
    Tag(('0008','1072')): 'X/D', # Operator Identification Sequence
    Tag(('0008','1070')): 'X/Z/D', # Operator's Name
    Tag(('0040','2010')): 'X', # Order Callback Phone Number
    Tag(('0040','2011')): 'X', # Order Callback Telecom Information
    Tag(('0040','2008')): 'X', # Order Entered By
    Tag(('0040','2009')): 'X', # Order Enterer's Location
    Tag(('0400','0561')): 'X', # Original Attributes Sequence
    Tag(('0010','1000')): 'X', # Other Patient IDs
    Tag(('0010','1002')): 'X', # Other Patient IDs Sequence
    Tag(('0010','1001')): 'X', # Other Patient Names
    # TODO - Come back to Overlay Comments and Overlay Data (pydicom repeaters dict)
    Tag(('0008','0024')): 'X', # Overlay Date
    Tag(('0008','0034')): 'X', # Overlay Time
    Tag(('300A','0760')): 'D', # Override Date/Time
    Tag(('0028','1199')): 'U', # Palette Color Lookup Table UID
    Tag(('0040','A07A')): 'X', # Participant Sequence
    Tag(('0040','A082')): 'Z', # Participation Date/Time
    Tag(('0010','1040')): 'X', # Patient Address
    Tag(('0010','1010')): 'X', # Patient's Age
    Tag(('0010','0030')): 'Z', # Patient's Birth Date
    Tag(('0010','1005')): 'X', # Patient's Birth Name
    Tag(('0010','0032')): 'X', # Patient's Birth Time
    Tag(('0038','0400')): 'X', # Patient's Institution Residence
    Tag(('0010','0050')): 'X', # Pataient's Insurance Plan Code Sequence
    Tag(('0010','1060')): 'X', # Patient's Mother's Birth Name
    Tag(('0010','0010')): 'Z', # Patient's Name
    Tag(('0010','0101')): 'X', # Patient's Primary Language Code Sequence
    Tag(('0010','0102')): 'X', # Patient's Primary Language Modifier Code Seq
    Tag(('0010','21F0')): 'X', # Patient's Religious Preference
    Tag(('0010','0040')): 'Z', # Patient's Sex
    Tag(('0010','2203')): 'X/Z', # Patient's Sex Neutered
    Tag(('0010','1020')): 'X', # Patient's Size
    Tag(('0010','2155')): 'X', # Patient's Telecom Information
    Tag(('0010','2154')): 'X', # Patient's Telephone Numbers
    Tag(('0010','1030')): 'X', # Patient's Weight
    Tag(('0010','4000')): 'X', # Patient Comments
    Tag(('0010','0020')): 'Z', # Patient ID
    Tag(('300A','0794')): 'X', # Patient Setup Photo Description
    Tag(('300A','0650')): 'U', # Patient Setup UID
    Tag(('0038','0500')): 'X', # Patient State
    Tag(('0040','1004')): 'X', # Patient Transport Arrangements
    Tag(('300A','0792')): 'X', # Patient Treatment Preparation Method Descrip
    Tag(('300A','078E')): 'X', # Patient Treatment Prep Procedure Param Descrip
    Tag(('0040','0243')): 'X', # Performed Location
    Tag(('0040','0254')): 'X', # Performed Procedure Step Description
    Tag(('0040','0250')): 'X', # Performed Procedure Step End Date
    Tag(('0040','4051')): 'X', # Performed Procedure Step End Date/Time
    Tag(('0040','0251')): 'X', # Performed Procedure Step End Time
    Tag(('0040','0253')): 'X', # Performed Procedure Step ID
    Tag(('0040','0244')): 'X', # Performed Procedure Step Start Date
    Tag(('0040','4050')): 'X', # Performed Procedure Step Start Date/Time
    Tag(('0040','0245')): 'X', # Performed Procedure Step Start Time
    Tag(('0040','0241')): 'X', # Performed Station AE Title
    Tag(('0040','4030')): 'X', # Performed Station Geographic Loc Code Seq
    Tag(('0040','0242')): 'X', # Performed Station Name
    Tag(('0040','4028')): 'X', # Performed Station Name Code Sequence
    Tag(('0008','1050')): 'X', # Performing Physician's Name
    Tag(('0008','1052')): 'X', # Performing Physician Identification Sequence
    Tag(('0040','1102')): 'X', # Person's Address
    Tag(('0040','1104')): 'X', # Person's Telecom Information
    Tag(('0040','1103')): 'X', # Person's Telephone Numbers
    Tag(('0040','1101')): 'D', # Person Identification Code Sequence
    Tag(('0040','A123')): 'D', # Person Name
    Tag(('0008','1048')): 'X', # Physician(s) of Record
    Tag(('0008','1049')): 'X', # Physician(s) of Record Identification Sequence
    Tag(('0008','1062')): 'X', # Physician(s) Reading Study Identification Seq
    Tag(('4008','0114')): 'X', # Physician Approving Interpretation
    Tag(('0040','2016')): 'Z', # Placer Order Number / Imaging Service Request
    Tag(('0018','1004')): 'X', # Plate ID
    Tag(('0010','21C0')): 'X', # Pregnancy Status
    Tag(('0040','0012')): 'X', # Pre-Medication
    Tag(('300A','000E')): 'X', # Prescription Description
    Tag(('3010','007B')): 'Z', # Prescription Notes
    Tag(('3010','0081')): 'Z', # Prescription Notes Sequence
    Tag(('0070','0082')): 'X', # Presentation Creation Date
    Tag(('0070','0083')): 'X', # Presentation Creation Time
    Tag(('0070','1101')): 'U', # Presentation Display Collection UID
    Tag(('0070','1102')): 'U', # Presentation Sequence Collection UID
    Tag(('3010','0061')): 'X', # Prior Treatment Dose Description
    Tag(('0040','4032')): 'X', # Procedure Step Cancellation Date/Time
    Tag(('0044','000B')): 'X', # Product Expiration Date/Time
    Tag(('0018','1030')): 'X/D', # Protocol Name
    Tag(('0008','1088')): 'X', # Pyramid Description
    Tag(('0020','0027')): 'X', # Pyramid Label
    Tag(('0008','0019')): 'U', # Pyramid UID
    Tag(('300A','0619')): 'D', # Radiation Dose Identification Label
    Tag(('300A','0623')): 'D', # Radiation Dose In-Vivo Measurement Label
    Tag(('300A','067D')): 'Z', # Radiation Generation Mode Description
    Tag(('300A','067C')): 'D', # Radiation Generation Mode Label
    Tag(('0018','1078')): 'X', # Radiopharmaceutical Start Date/Time
    Tag(('0018','1072')): 'X', # Radiopharmaceutical Start Time
    Tag(('0018','1079')): 'X', # Radiopharmaceutical Stop Date/Time
    Tag(('0018','1073')): 'X', # Radiopharmaceutical Stop Time
    Tag(('300C','0113')): 'X', # Reason for Omission Description
    Tag(('0040','100A')): 'X', # Reason for Requested Procedure Code Sequence
    Tag(('0032','1030')): 'X', # Reason for Study
    Tag(('3010','005C')): 'Z', # Reason for Superseding
    Tag(('0400','0565')): 'D', # Reason for the Attribute Modification
    Tag(('0040','2001')): 'X', # Reason for the Imaging Service Request
    Tag(('0040','1002')): 'X', # Reason for the Requested Procedure
    Tag(('0032','1066')): 'X', # Reason for Visit
    Tag(('0032','1067')): 'X', # Reason for Visit Code Sequence
    Tag(('300A','073A')): 'D', # Recorded RT Control Point Date/Time
    Tag(('3010','000B')): 'U', # Referenced Conceptual Volume UID
    Tag(('0040','A13A')): 'D', # Referenced Date/Time
    Tag(('0400','0402')): 'X', # Referenced Digital Signature Sequence
    Tag(('300A','0083')): 'U', # Referenced Dose Reference UID
    Tag(('3010','0031')): 'U', # Referenced Fiducials UID
    Tag(('3006','0024')): 'U', # Referenced Frame of Reference UID
    Tag(('0040','4023')): 'U', # Referenced Gen Purpose Scheduled Procedure Step Transaction UID
    Tag(('0008','1140')): 'X/Z/U', # Referenced Image Sequence
    Tag(('0040','A172')): 'U', # Referenced Observation UID (Trial)
    Tag(('0038','0004')): 'X', # Referenced Patient Alias Sequence
    Tag(('0010','1100')): 'X', # Referenced Patient Photo Sequence
    Tag(('0008','1120')): 'X', # Referenced Patient Sequence
    Tag(('0008','1111')): 'X/Z/D', # Referenced Performed Procedure Step Seq
    Tag(('0400','0403')): 'X', # Referenced SOP Instance MAC Sequence
    Tag(('0008','1155')): 'U', # Referenced SOP Instance UID
    Tag(('0004','1511')): 'U', # Referenced SOP Instance UID in File
    Tag(('0008','1110')): 'X/Z', # Referenced Study Sequence
    Tag(('300A','0785')): 'U', # Referenced Treatment Position Group UID
    Tag(('0008','0092')): 'X', # Referring Physician's Address
    Tag(('0008','0090')): 'Z', # Referring Physician's Name
    Tag(('0008','0094')): 'X', # Referring Physician's Telephone Numbers
    Tag(('0008','0096')): 'X', # Referring Physician Identification Sequence
    Tag(('0010','2152')): 'X', # Region of Residence
    Tag(('3006','00C2')): 'U', # Related Frame of Reference UID
    Tag(('0040','0275')): 'X', # Request Attributes Sequence
    Tag(('0032','1070')): 'X', # Requested Contrast Agent
    Tag(('0040','1400')): 'X', # Requested Procedure Comments
    Tag(('0032','1060')): 'X/Z', # Requested Procedure Description
    Tag(('0040','1001')): 'X', # Requested Procedure ID
    Tag(('0040','1005')): 'X', # Requsted Procedure Location
    Tag(('0018','9937')): 'X', # Requested Series Description
    Tag(('0000','1001')): 'U', # Requested SOP Instance UID
    Tag(('0032','1032')): 'X', # Requesting Physician
    Tag(('0032','1033')): 'X', # Requesting Service
    Tag(('0018','9185')): 'X', # Respiratory Motion Compensation Technique Description
    Tag(('0010','2299')): 'X', # Responsible Organization
    Tag(('0010','2297')): 'X', # Responsible Person
    Tag(('4008','4000')): 'X', # Results Comments
    Tag(('4008','0118')): 'X', # Results Distribution List Sequence
    Tag(('4008','0040')): 'X', # Results ID
    Tag(('4008','0042')): 'X', # Results ID Issuer
    Tag(('300E','0004')): 'Z', # Review Date
    Tag(('300E','0008')): 'X/Z', # Reviewer Name
    Tag(('300E','0005')): 'Z', # Review Time
    Tag(('3006','0028')): 'X', # ROI Description
    Tag(('3006','0038')): 'X', # ROI Generation Description
    Tag(('3006','00A6')): 'Z', # ROI Interpreter
    # Tag(('3006','0026')): 'Z', # ROI Name - Overriding this, need to keep
    Tag(('3006','0088')): 'X', # ROI Observation Description
    # Tag(('3006','0085')): 'X', # ROI Observation Label
    Tag(('300A','0615')): 'Z', # RT Accessory Device Slot ID
    Tag(('300A','0611')): 'Z', # RT Accessory Holder Slot ID
    Tag(('3010','005A')): 'Z', # RT Physician Intent Narrative
    Tag(('300A','0006')): 'X/D', # RT Plan Date
    Tag(('300A','0004')): 'X', # RT Plan Description
    Tag(('300A','0002')): 'D', # RT Plan Label
    Tag(('300A','0003')): 'X', # RT Plan Name
    Tag(('300A','0007')): 'X/D', # RT Plan Time
    Tag(('3010','0054')): 'D', # RT Prescription Label
    Tag(('300A','062A')): 'D', # RT Tolerance Set Label
    Tag(('3010','0056')): 'X/D', # RT Treatment Approach Label
    Tag(('3010','003B')): 'U', # RT Treatment Phase UID
    Tag(('3008','0162')): 'D', # Safe Position Exit Date
    Tag(('3008','0164')): 'D', # Safe Position Exit Time
    Tag(('3008','0166')): 'D', # Safe Position Return Date
    Tag(('3008','0168')): 'D', # Safe Position Return Time
    Tag(('0038','001A')): 'X', # Scheduled Admission Date
    Tag(('0038','001B')): 'X', # Scheduled Admission Time
    Tag(('0038','001C')): 'X', # Scheduled Discharge Date
    Tag(('0038','001D')): 'X', # Scheduled Discharge Time
    Tag(('0040','4034')): 'X', # Scheduled Human Performers Sequence
    Tag(('0038','001E')): 'X', # Scheduled Patient Institution Residence
    Tag(('0040','0006')): 'X', # Scheduled Performing Physician's Name
    Tag(('0040','000B')): 'X', # Scheduled Performing Physician Ident Seq
    Tag(('0040','0007')): 'X', # Scheduled Procedure Step Description
    Tag(('0040','0004')): 'X', # Scheduled Procedure Step End Date
    Tag(('0040','0005')): 'X', # Scheduled Procedure Step End Time
    Tag(('0040','4008')): 'X', # Scheduled Procedure Step Expiration Date/Time
    Tag(('0040','0009')): 'X', # Scheduled Procedure Step ID
    Tag(('0040','0011')): 'X', # Scheduled Procedure Step Location
    Tag(('0040','4010')): 'X', # Scheduled Procedure Step Modification Date/Time
    Tag(('0040','0002')): 'X', # Scheduled Procedure Step Start Date
    Tag(('0040','4005')): 'X', # Scheduled Procedure Step Start Date/Time
    Tag(('0040','0003')): 'X', # Scheduled Procedure Step Start Time
    Tag(('0040','0001')): 'X', # Scheduled Station AE Title
    Tag(('0040','4027')): 'X', # Scheduled Station Geograph Loc Code Seq
    Tag(('0040','0010')): 'X', # Scheduled Station Name
    Tag(('0040','4025')): 'X', # Scheduled Station Name Code Sequence
    Tag(('0032','1020')): 'X', # Scheduled Study Location
    Tag(('0032','1021')): 'X', # Scheduled Study Location AE Title
    Tag(('0032','1000')): 'X', ## Scheduled Study Start Date
    Tag(('0032','1001')): 'X', # Scheduled Study Start Time
    Tag(('0032','1010')): 'X', # Scheduled Study Stop Date
    Tag(('0032','1011')): 'X', # Scheduled Study Stop Time
    Tag(('0072','005F')): 'D', # Selector AS Value
    Tag(('0072','0061')): 'D', # Selector DA Value
    Tag(('0072','0063')): 'D', # Selector DT Value
    Tag(('0072','0066')): 'D', # Selector LO Value
    Tag(('0072','0068')): 'D', # Selector LT Value
    Tag(('0072','0065')): 'D', # Selector OB Value
    Tag(('0072','006A')): 'D', # Selector PN Value
    Tag(('0072','006C')): 'D', # Selector SH Value
    Tag(('0072','006E')): 'D', # Selector ST Value
    Tag(('0072','006B')): 'D', # Selector TM Value
    Tag(('0072','006D')): 'D', # Selector UN Value
    Tag(('0072','0071')): 'D', # Selector UR Value
    Tag(('0072','0070')): 'D', # Selector UT Value
    Tag(('0008','0021')): 'X/D', # Series Date
    Tag(('0008','103E')): 'X', # Series Description
    Tag(('0020','000E')): 'U', # Series Instance UID
    Tag(('0008','0031')): 'X/D', # Series Time
    Tag(('0038','0062')): 'X', # Service Episode Description
    Tag(('0038','0060')): 'X', # Service Episode ID
    Tag(('300A','01B2')): 'X', # Setup Technique Description
    Tag(('300A','01A6')): 'X', # Shielding Device Description
    Tag(('0040','06FA')): 'X', # Slide Identifier
    Tag(('0010','21A0')): 'X', # Smoking Status
    Tag(('0100','0420')): 'X', # SOP Authorization Date/Time
    Tag(('0008','0018')): 'U', # SOP Instance UID
    Tag(('3010','0015')): 'U', # Source Conceptual Volume UID
    Tag(('0018','936A')): 'D', # Source End Date/Time
    Tag(('0034','0005')): 'D', # Source Identifier
    Tag(('0008','2112')): 'X/Z/U', # Source Image Sequence
    Tag(('300A','0216')): 'X', # Source Manufacturer
    Tag(('0400','0564')): 'Z', # Source of Previous Values
    Tag(('3008','0105')): 'X/Z', # Source Serial Number
    Tag(('0018','9369')): 'D', # Source Start Date/Time
    Tag(('300A','022C')): 'D', # Source Strength Reference Date
    Tag(('300A','022E')): 'D', # Source Strength Reference Time
    Tag(('0038','0050')): 'X', # Special Needs
    Tag(('0040','050A')): 'X', # Specimen Accession Number
    Tag(('0040','0602')): 'X', # Specimen Detailed Description
    Tag(('0040','0551')): 'D', # Specimen Identifier
    Tag(('0040','0610')): 'Z', # Specimen Preparation Sequence
    Tag(('0040','0600')): 'X', # Specimen Short Description
    Tag(('0040','0554')): 'U', # Specimen UID
    Tag(('0018','9516')): 'X/D', # Start Acquisition Date/Time
    Tag(('0008','1010')): 'X/Z/D', # Station Name
    Tag(('0088','0140')): 'U', # Storage Media File-Set UID
    Tag(('3006','0008')): 'Z', # Structure Set Date
    Tag(('3006','0006')): 'X', # Structure Set Description
    Tag(('3006','0002')): 'D', # Structure Set Label
    Tag(('3006','0004')): 'X', # Structure Set Name
    Tag(('3006','0009')): 'Z', # Structure Set Time
    Tag(('0032','1040')): 'X', # Study Arrival Date
    Tag(('0032','1041')): 'X', # Study Arrival Time
    Tag(('0032','4000')): 'X', # Study Comments
    Tag(('0032','1050')): 'X', # Study Completion Date
    Tag(('0032','1051')): 'X', # Study Completion Time
    Tag(('0008','0020')): 'Z', # Study Date
    Tag(('0008','1030')): 'X', # Study Description
    Tag(('0020','0010')): 'Z', # Study ID
    Tag(('0032','0012')): 'X', # Study ID Issuer
    Tag(('0020','000D')): 'U', # Study Instance UID
    Tag(('0032','0034')): 'X', # Study Read Date
    Tag(('0032','0035')): 'X', # Study Read Time
    Tag(('0008','0030')): 'Z', # Study Time
    Tag(('0032','0032')): 'X', # Study Verified Date
    Tag(('0032','0033')): 'X', # Study Verified Time
    Tag(('0044','0010')): 'X', # Substance Administration Date/Time
    Tag(('0020','0200')): 'U', # Synchronization Frame of Reference UID
    Tag(('0018','2042')): 'U', # Target UID
    Tag(('0040','A354')): 'X', # Telephone Number (Trial)
    Tag(('0040','DB0D')): 'U', # Template Extension Creator UID
    Tag(('0040','DB0C')): 'U', # Template Extension Organizer UID
    Tag(('0040','DB07')): 'X', # Template Local Version
    Tag(('0040','DB06')): 'X', # Template Version
    Tag(('4000','4000')): 'X', # Text Comments
    Tag(('2030','0020')): 'X', # Text String
    Tag(('0040','A122')): 'D', # Time
    Tag(('0040','A112')): 'X', # Time of Document or Verbal Transaction (Trial)
    Tag(('0018','1201')): 'X', # Time of Last Calibration
    Tag(('0018','700E')): 'X/D', # Time of Last Detector Calibration
    Tag(('0018','1014')): 'X', # Time of Secondary Capture
    Tag(('0008','0201')): 'X', # Timezone Offset from UTC
    Tag(('0088','0910')): 'X', # Topic Author
    Tag(('0088','0912')): 'X', # Topic Keywords
    Tag(('0088','0906')): 'X', # Topic Subject
    Tag(('0088','0904')): 'X', # Topic Title
    Tag(('0062','0021')): 'U', # Tracking UID
    Tag(('0008','1195')): 'U', # Transaction UID
    Tag(('0018','5011')): 'X', # Transducer Identification Sequence
    Tag(('3008','0024')): 'D', # Treatment Control Point Time
    Tag(('3008','0250')): 'X/D', # Treatment Date
    Tag(('300A','00B2')): 'X/Z', # Treatment Machine Name
    Tag(('300A','0608')): 'D', # Treatment Position Group Label
    Tag(('300A','0609')): 'U', # Treatment Position Group UID
    Tag(('300A','0700')): 'U', # Treatment Session UID
    Tag(('3010','0077')): 'X/D', # Treatment Site
    Tag(('300A','000B')): 'X', # Treatment Sites
    Tag(('3010','007A')): 'Z', # Treatment Technique Notes
    Tag(('3008','0251')): 'X/D', # Treatment Time
    Tag(('300A','0736')): 'D', # Treatment Tolerance Violation Date/Time
    Tag(('300A','0734')): 'D', # Treatment Tolerance Violation Description
    Tag(('0018','100A')): 'X', # UDI Sequence
    Tag(('0040','A124')): 'U', # UID
    Tag(('0018','1009')): 'X', # Unique Device Identifier
    Tag(('3010','0033')): 'D', # User Content Label
    Tag(('3010','0034')): 'D', # User Content Long Label
    Tag(('0040','A352')): 'X', # Verbal Source (Trial)
    Tag(('0040','A358')): 'X', # Verbal Source Identifier Code Seq (Trial)
    Tag(('0040','A030')): 'D', # Verification Date/Time
    Tag(('0040','A088')): 'Z', # Verifying Observer Identification Code Seq
    Tag(('0040','A075')): 'D', # Verifying Observer Name
    Tag(('0040','A073')): 'D', # Verifying Observer Sequence
    Tag(('0040','A027')): 'D', # Verifying Organization
    Tag(('0038','4000')): 'X', # Visit Comments
    Tag(('003A','0329')): 'X', # Waveform Filter Description
    Tag(('0018','9371')): 'D', # X-Ray Detector ID
    Tag(('0018','9373')): 'X', # X-Ray Detector Label
    Tag(('0018','9367')): 'D', # X-Ray Source ID
    }

"""
Variable Tags: These are tags whose prescribed procedure changes depending
on the IOD modules for a given modality.
    Tag(('0040','0555')): 'X/Z', # Acquisition Context Sequence
    Tag(('0008','0022')): 'X/Z', # Acquisition Date
    Tag(('0008','002A')): 'X/Z/D', # Acquisition Date/Time
    Tag(('0018','1400')): 'X/D', # Acquisition Device Processing Description
    Tag(('0008','0032')): 'X/Z', # Acquisition Time
    Tag(('2200','0005')): 'X/Z', # Barcode Value
    Tag(('0070','0084')): 'Z/D', # Content Creator's Name
    Tag(('0008','0023')): 'Z/D', # Content Date
    Tag(('0008','0033')): 'Z/D', # Content Time
    Tag(('0018','0010')): 'Z/D', # Contrast/Bolus Agent
    Tag(('0018','700C')): 'X/D', # Date of Last Detector Calibration
    Tag(('0018','700A')): 'X/D', # Detector ID
    Tag(('0018','1000')): 'X/Z/D', # Device Serial Number
    Tag(('0018','9517')): 'X/D', # End Acquisition Date/Time
    Tag(('3008','0054')): 'X/D', # First Treatment Date
    Tag(('0008','0012')): 'X/D', # Instance Creation Date
    Tag(('0008','0013')): 'X/Z/D', # Instance Creation Time
    Tag(('0008','0082')): 'X/Z/D', # Institution Code Sequence
    Tag(('0008','0080')): 'X/Z/D', # Institution Name
    Tag(('0018','9919')): 'Z/D', # Instruction Performed Date/Time
    Tag(('3010','004D')): 'X/D', # Intended Phase End Date
    Tag(('3010','004C')): 'X/D', # Intended Phase Start Date
    Tag(('2200','0002')): 'X/Z', # Label Text
    Tag(('3008','0056')): 'X/D', # Most Recent Treatment Date
    Tag(('0040','A032')): 'X/D', # Observation Date/Time
    Tag(('0008','1072')): 'X/D', # Operator Identification Sequence
    Tag(('0008','1070')): 'X/Z/D', # Operator's Name
    Tag(('0010','2203')): 'X/Z', # Patient's Sex Neutered
    Tag(('0018','1030')): 'X/D', # Protocol Name
    Tag(('0008','1140')): 'X/Z/U', # Referenced Image Sequence
    Tag(('0008','1111')): 'X/Z/D', # Referenced Performed Procedure Step Seq
    Tag(('0008','1110')): 'X/Z', # Referenced Study Sequence
    Tag(('0032','1060')): 'X/Z', # Requested Procedure Description
    Tag(('300E','0008')): 'X/Z', # Reviewer Name
    Tag(('300A','0006')): 'X/D', # RT Plan Date
    Tag(('300A','0007')): 'X/D', # RT Plan Time
    Tag(('3010','0056')): 'X/D', # RT Treatment Approach Label
    Tag(('0008','0021')): 'X/D', # Series Date
    Tag(('0008','0031')): 'X/D', # Series Time
    Tag(('0008','2112')): 'X/Z/U', # Source Image Sequence
    Tag(('3008','0105')): 'X/Z', # Source Serial Number
    Tag(('0018','9516')): 'X/D', # Start Acquisition Date/Time
    Tag(('0008','1010')): 'X/Z/D', # Station Name
    Tag(('0018','700E')): 'X/D', # Time of Last Detector Calibration
    Tag(('3008','0250')): 'X/D', # Treatment Date
    Tag(('300A','00B2')): 'X/Z', # Treatment Machine Name
    Tag(('3010','0077')): 'X/D', # Treatment Site
    Tag(('3008','0251')): 'X/D', # Treatment Time
"""
# MODULE LISTS FOR IOD USE
# This provides relevant overrides for the above mentioned variable fields
# We only include module fields that cover the previously listed variable tags
# since non-variable tags can just be read from basic profile
patient = {}
clinical_trial_subject = {}
general_study = {
    Tag(('0008','1110')): 'X', # Referenced Study Sequence - Type 3
    }
patient_study = {}
clinical_trial_study = {}
general_series = {
    Tag(('0008','0021')): 'X', # Series Date - Type 3
    Tag(('0008','0031')): 'X', # Series Time - Type 3
    }
clinical_trial_series = {}
frame_of_reference = {}
synchronization = {}
general_equipment = {
    Tag(('0008','0080')): 'X', # Institution Name - Type 3
    }
general_acquisition = {
    Tag(('0008','0022')): 'X', # Acquisition Date - Type 3
    Tag(('0008','002A')): 'X', # Acquisition Date/Time - Type 3
    Tag(('0008','0032')): 'X', # Acquisition Time - Type 3
    }
general_image = {
    Tag(('0008','0023')): 'Z', # Content Date - Type 2
    Tag(('0008','0033')): 'Z', # Content Time - Type 2
    }
general_reference = {
    Tag(('0008','1140')): 'X', # Referenced Image Sequence - Type 3
    }
image_plane = {}
image_pixel = {}
contrast_bolus = {
    Tag(('0018','0010')): 'Z', # Contrast/Bolus Agent - Type 2
    }
device = {
    Tag(('0018','1000')): 'X', # Device Serial Number - Type 3
    }
specimen = {}
ct_image = {}
mr_image = {}
multienergy_ct_image = {}
overlay_plane = {}
voi_lut = {}
sop_common = {
    Tag(('0008','0012')): 'X', # Instance Creation Date - Type 3
    Tag(('0008','0013')): 'X', # Instance Creation Time - Type 3
    }
common_instance_reference = {}
rt_series = {
    Tag(('0008','0021')): 'X', # Series Date - Type 3
    Tag(('0008','0031')): 'X', # Series Time - Type 3
    Tag(('0008','1070')): 'Z', # Operator's Name - Type 2
    Tag(('0008','1072')): 'X', # Operator Identification Sequence - Type 3
    }
multiframe = {}
multiframe_overlay = {}
modality_lut = {}
rt_dose = {
    Tag(('0008','0023')): 'Z', # Content Date - Type 3, but option is 'Z/D'
    Tag(('0008','0033')): 'Z', # Content Time - Type 3, but option is 'Z/D'
    }
rt_dvh = {}
structure_set = {}
roi_contour = {}
rt_dose_roi = {}
frame_extraction = {}
rt_roi_observations = {}
approval = {
    Tag(('300E','0008')): 'Z', # Reviewer Name - Type 2C
    }
rt_general_plan = {
    Tag(('300A','0006')): 'D', # RT Plan Date - Type 2, but 'X/D'
    Tag(('300A','0007')): 'D', # RT Plan Time - Type 2, but 'X/D'
    Tag(('3010','0077')): 'X', # Treatment Site - Type 3
    }
rt_prescription = {}
rt_tolerance_tables = {}
rt_patient_setup = {}
rt_fraction_scheme = {}
rt_beams = {
    Tag(('300A','00B2')): 'Z', # Treatment Machine Name - Type 2
    Tag(('0008','0080')): 'X', # Institution Name - Type 3
    }
rt_brachy_application_setups = {
    Tag(('300A','00B2')): 'Z', # Treatment Machine Name - Type 2
    Tag(('0008','0080')): 'X', # Institution Name - Type 3
    Tag(('3008','0105')): 'X', # Source Serial Number - Type 3
    }
pet_series = {
    Tag(('0008','0021')): 'D', # Series Date - Type 1
    Tag(('0008','0031')): 'D', # Series Time - Type 1
    }
pet_isotope = {}
pet_multigated_acquisition = {}
nmpet_patient_orientation = {}
pet_image = {
    Tag(('0008','0022')): 'X', # Acquisition Date - Type 3
    Tag(('0008','0032')): 'X', # Acquisition Time - Type 3
    }
acquisition_context = {
    Tag(('0040','0555')): 'Z', # Acquisition Context Sequence - Type 2
    }

# IOD lists - lists of which module dictionaries are valid for which IODs
ct_image_iod = [
    patient, clinical_trial_subject, general_study, patient_study,
    clinical_trial_study, general_series, clinical_trial_series,
    frame_of_reference, synchronization, general_equipment, 
    general_acquisition, general_image, general_reference, image_plane,
    image_pixel, contrast_bolus, device, specimen, multienergy_ct_image,
    overlay_plane, voi_lut, sop_common, common_instance_reference
    ]
mr_image_iod = [
    patient, clinical_trial_subject, general_study, patient_study,
    clinical_trial_study, general_series, clinical_trial_series,
    frame_of_reference, general_equipment, general_acquisition, general_image,
    general_reference, image_plane, image_pixel, contrast_bolus, device,
    specimen, mr_image, overlay_plane, voi_lut, sop_common, 
    common_instance_reference
    ]
rt_dose_iod = [
    patient, clinical_trial_subject, general_study, patient_study,
    clinical_trial_study, rt_series, clinical_trial_series, frame_of_reference,
    general_equipment, general_image, image_plane, image_pixel, multiframe,
    overlay_plane, multiframe_overlay, modality_lut, rt_dose, rt_dvh,
    structure_set, roi_contour, rt_dose_roi, sop_common, 
    common_instance_reference, frame_extraction
    ]
rt_structure_set_iod = [
    patient, clinical_trial_subject, general_study, patient_study,
    clinical_trial_study, rt_series, clinical_trial_series, frame_of_reference,
    general_equipment, structure_set, roi_contour, rt_roi_observations,
    approval, general_reference, sop_common, common_instance_reference
    ]
rt_plan_iod = [
    patient, clinical_trial_subject, general_study, patient_study,
    clinical_trial_study, rt_series, clinical_trial_series, frame_of_reference,
    general_equipment, rt_general_plan, rt_prescription, rt_tolerance_tables,
    rt_patient_setup, rt_fraction_scheme, rt_beams,
    rt_brachy_application_setups, approval, general_reference, sop_common,
    common_instance_reference
    ]
pet_image_iod = [
    patient, clinical_trial_subject, general_study, patient_study,
    clinical_trial_study, general_series, clinical_trial_series,
    pet_series, pet_isotope, pet_multigated_acquisition, 
    nmpet_patient_orientation, frame_of_reference, synchronization,
    general_equipment, general_acquisition, general_image, general_reference,
    image_plane, image_pixel, device, specimen, pet_image, overlay_plane,
    voi_lut, acquisition_context, sop_common, common_instance_reference
    ]