#!/usr/bin/env bash
SCPADDRESS="127.0.0.1"
SCPPORT=1124
AEARIA="AriaDatabase"

for patientid in $(<idlist.txt); do
    movescu -S -aec ${AEARIA} -k 0008,0052=STUDY -k 0010,0020=${patientid} -k 0008,0061=CT ${SCPADDRESS} ${SCPPORT}
done