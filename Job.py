from model.VGG import Partition_Model_VGG
class Job:
    def __init__(self,ID,batch_size,Epoch):
        self.ID=ID
        self.batch_size=batch_size
        self.Epoch=Epoch
        partition_lists=Partition()
        self.Sub_Model_List=Partition_Model_VGG(partition_lists,batch_size,Epoch)

def Generate_job_stream(Job_number):
    job_stream=[]
    for i in range(Job_number):
        job=Job(i,64,10)
        job_stream.append(job)
    return job_stream