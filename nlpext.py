from fpdf import FPDF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

class Reporter:
    def __init__(self,csv,results,datpoints,confidence):
        self.csv = csv
        self.results = results
        self.datpoints = datpoints
        self.confidence = confidence
        
    def makePDF(self):
        for i in range(len(results)):
            cur = self.csv.loc[i]
            d = str(cur["date"])
            d = d.replace("_"," ",1)
            d = d.replace("_",":")
            a = cur["age"]
            s = cur["sex"]
            scno = cur["scanid"]
            subid = cur["subject"]
            scty = cur["type"]
            length = cur["len"]
            thick = tuple(cur["thick"])
            title = "Textual Report for given fMRI for Alzheimer's Disease:"
            dat = 'Date: '+str(d)
            age = "Age: "+str(a)
            sex = "Sex: "+str(s)
            scan_no = "Scan ID: "+str(scno)
            subj_id = "Subject ID: "+str(subid)
            scan_type = "Scan Type: "+str(scty)
            proto = "The slices were taken with dimensions of "+str(thick[0])+", "+str(thick[1])+" and "+str(thick[2])+" in the axial, saggital and coronal axes respectively. There are "+str(length)+" time points out of which "+str(datpoints)+" have been selected for optical flow comparison. The optical flow was taken in mirror fashion. The fMRI image was of resting state and thus complies with the native training method of the model. The neural network is the default 6 layer network."
            if result[i]=="ADRS":
                results = "Alzheimer's Disease with a stage of Dementia suspected with confidence of "+str(self.confidence[i].max())+"% at the current average accuracy of 73.91% of the model. Other possibility is that patient is Cognitively Normal which is suspected at confidence of "+str(self.confidence[i].min())+"% hence kept secondary."
                labels = labels = ["Alzheimer's Dementia","Cognitively Normal"]
            elif result[i]=="CNRS":
                results = "Cognitively the patient is suspected to be normal with confidence of "+str(self.confidence[i].max())+"% at the current average accuracy of 73.91% of the model. Other possibility is Alzheimer's Disease with Dementia which was suspected at confidence of "+str(self.confidence[i].min())+"% hence kept secondary."
                labels = ["Cognitively Normal","Alzheimer's Dementia"]
            else:
                results = "Invalid value found."
            pdf = FPDF('P', 'in', 'A4')
            pdf.add_page('P')
            pdf.set_font('Times', 'B', 18)
            effective_page_width = pdf.w - 2*pdf.l_margin
            pdf.cell(0, 0, title,align="C")
            pdf.set_font('Times', '', 15)
            ybefore = pdf.get_y()
            ybefore+=0.45
            pdf.set_fill_color(224,240,255)
            pdf.set_xy(pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width/2, 0.45, name, fill=1, border=1, align='J')
            pdf.set_fill_color(255, 255, 255)
            pdf.set_xy(effective_page_width/2 + pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width/2, 0.45, age, fill=1, border=1, align='J')
            pdf.set_fill_color(255, 255, 255)
            ybefore+=0.45
            pdf.set_xy(pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width/2, 0.45, sex, fill=1, border=1, align='J')
            pdf.set_fill_color(224, 240, 255)
            pdf.set_xy(effective_page_width/2 + pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width/2, 0.45, scan_no, fill=1, border=1, align='J')
            pdf.set_fill_color(224, 240, 255)
            ybefore+=0.45
            pdf.set_xy(pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width/2, 0.45, subj_id, fill=1, border=1, align='J')
            pdf.set_fill_color(255, 255, 255)
            pdf.set_xy(effective_page_width/2 + pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width/2, 0.45, scan_type, fill=1, border=1, align='J')

            plt.rcdefaults()
            fig, ax = plt.subplots()
            conf = [self.confidence[i].max(),self.confidence[i].min()]
            y_pos = np.arange(len(labels))
            ax.barh(conf, performance, xerr=error, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(people)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel("Confidence")
            ax.set_title("Alzheimer's Report from fMRI")
            savefig('reports/'+subid+"_"+date+".png", bbox_inches='tight')
            
            pdf.image('reports/'+subid+"_"+date+".png",effective_page_width/4,ybefore+0.25,effective_page_width/2,3.5)
            
            ybefore+=4
            pdf.set_xy(pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width,0.3,"Protocol: "+proto,align="J")
            ybefore+=1.8
            pdf.set_xy(pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width,0.3,"Results and Observations: "+results,align="J")
            ybefore+=1.2
            pdf.set_xy(pdf.l_margin, ybefore)
            pdf.multi_cell(effective_page_width,0.3,"For Doctor's eyes only.\nAll these readings are not to be determined as concrete diagnosis. This is an alpha test and has to be verified carefully. These readings can act as a diagnostic support tool but are not to be interpreted as an indicative diagnosis.",align="J")

            pdf.output(('reports/'+subid+"_"+date+".pdf"), 'F')
