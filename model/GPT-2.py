from code.vlayer0 import vLayer0
from code.vlayer1 import vLayer1
from code.vlayer2 import vLayer2
from code.vlayer3 import vLayer3
from code.vlayer4 import vLayer4
from code.vlayer5 import vLayer5
from code.vlayer6 import vLayer6
from code.vlayer7 import vLayer7
from code.vlayer8 import vLayer8
from code.vlayer9 import vLayer9
from code.vlayer10 import vLayer10
from code.vlayer11 import vLayer11
from code.vlayer12 import vLayer12
from code.vlayer13 import vLayer13
from code.vlayer14 import vLayer14
from code.vlayer15 import vLayer15
from code.vlayer16 import vLayer16
from code.vlayer17 import vLayer17
from code.vlayer18 import vLayer18
from code.vlayer19 import vLayer19
from code.vlayer20 import vLayer20
from code.vlayer21 import vLayer21
from code.vlayer22 import vLayer22
from code.vlayer23 import vLayer23
from code.vlayer24 import vLayer24
from code.vlayer25 import vLayer25
from code.vlayer26 import vLayer26
from code.vlayer27 import vLayer27
from code.vlayer28 import vLayer28
from code.vlayer29 import vLayer29
from code.vlayer30 import vLayer30
from code.vlayer31 import vLayer31
from code.vlayer32 import vLayer32
from code.vlayer33 import vLayer33
from code.vlayer34 import vLayer34
from code.vlayer35 import vLayer35
from code.vlayer36 import vLayer36
from code.vlayer37 import vLayer37
from code.vlayer38 import vLayer38
from code.vlayer39 import vLayer39
from code.vlayer40 import vLayer40
from code.vlayer41 import vLayer41
from code.vlayer42 import vLayer42
from code.vlayer43 import vLayer43
from code.vlayer44 import vLayer44
from code.vlayer45 import vLayer45
from code.vlayer46 import vLayer46
from code.vlayer47 import vLayer47
from code.vlayer48 import vLayer48
from code.vlayer49 import vLayer49
from code.vlayer50 import vLayer50


class GPT_2_B1:
    def __init__(self,config):
        self.Layer0=vLayer0(config)
        self.Layer1=vLayer1(config)
        self.Layer2=vLayer2(config)
        self.Layer3=vLayer3(config)
        self.Layer4=vLayer4(config)
        self.Layer5=vLayer5(config)
        self.Layer6=vLayer6(config)
        self.Layer7=vLayer7(config)
        self.Layer8=vLayer8(config)
        self.Layer9=vLayer9(config)
        self.Layer10=vLayer10(config)
        self.Layer11=vLayer11(config)

    
    def forward(self,input):
        output=self.Layer0(input)
        output=self.Layer1(output)
        output=self.Layer2(output)
        output=self.Layer3(output)
        output=self.Layer4(output)
        output=self.Layer5(output)
        output=self.Layer6(output)
        output=self.Layer7(output)
        output=self.Layer8(output)
        output=self.Layer9(output)
        output=self.Layer10(output)
        output=self.Layer11(output)
        return output




class GPT_2_B2:
    def __init__(self,config):
        self.Layer12=vLayer12(config)
        self.Layer13=vLayer13(config)
        self.Layer14=vLayer14(config)
        self.Layer15=vLayer15(config)
        self.Layer16=vLayer16(config)
        self.Layer17=vLayer17(config)
        self.Layer18=vLayer18(config)
        self.Layer19=vLayer19(config)
        self.Layer20=vLayer20(config)
        self.Layer21=vLayer21(config)
        self.Layer22=vLayer22(config)
        self.Layer23=vLayer23(config)
        self.Layer24=vLayer24(config)

    def forward(self,input):
        output=self.Layer12(input)
        output=self.Layer13(output)
        output=self.Layer14(output)
        output=self.Layer15(output)
        output=self.Layer16(output)
        output=self.Layer17(output)
        output=self.Layer18(output)
        output=self.Layer19(output)
        output=self.Layer20(output)
        output=self.Layer21(output)
        output=self.Layer22(output)
        output=self.Layer23(output)
        output=self.Layer24(output)
        return output

class GPT_2_B3:
    def __init__(self,config):        
        self.Layer25=vLayer25(config)
        self.Layer26=vLayer26(config)
        self.Layer27=vLayer27(config)
        self.Layer28=vLayer28(config)
        self.Layer29=vLayer29(config)
        self.Layer30=vLayer30(config)
        self.Layer31=vLayer31(config)
        self.Layer32=vLayer32(config)
        self.Layer33=vLayer33(config)
        self.Layer34=vLayer34(config)
        self.Layer35=vLayer35(config)
        self.Layer36=vLayer36(config)
        self.Layer37=vLayer37(config)

    def forward(self,input):
        output=self.Layer25(input)
        output=self.Layer26(output)
        output=self.Layer27(output)
        output=self.Layer28(output)
        output=self.Layer29(output)
        output=self.Layer30(output)
        output=self.Layer31(output)
        output=self.Layer32(output)
        output=self.Layer33(output)
        output=self.Layer34(output)
        output=self.Layer35(output)
        output=self.Layer36(output)
        output=self.Layer37(output)
        return output
    

class GPT_2_B4:
    def __init__(self,config):
        self.Layer38=vLayer38(config)
        self.Layer39=vLayer39(config)
        self.Layer40=vLayer40(config)
        self.Layer41=vLayer41(config)
        self.Layer42=vLayer42(config)
        self.Layer43=vLayer43(config)
        self.Layer44=vLayer44(config)
        self.Layer45=vLayer45(config)
        self.Layer46=vLayer46(config)
        self.Layer47=vLayer47(config)
        self.Layer48=vLayer48(config)
        self.Layer49=vLayer49(config)
        self.Layer50=vLayer50(config)

    def forward(self,input):
        output=self.Layer38(input)
        output=self.Layer39(output)
        output=self.Layer40(output)
        output=self.Layer41(output)
        output=self.Layer42(output)
        output=self.Layer43(output)
        output=self.Layer44(output)
        output=self.Layer45(output)
        output=self.Layer46(output)
        output=self.Layer47(output)
        output=self.Layer48(output)
        output=self.Layer49(output)
        output=self.Layer50(output)
        return output