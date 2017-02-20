import csv

class CSVParse(object):

    def __init__(self):
        self.data = []

    def read(self, filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, dialect='excel', delimiter=';')
            for row in reader:
                if row == ['','','','','','','','']:
                    pass
                else:
                    newRow = []
                    for col in row:
                        col = col.replace(',', '.')
                        try:
                            col = float(col)
                        except:
                            pass
                        newRow.append(col)
                    self.data.append(newRow)
        return(self.data)

    def write(self, filename):
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for row in self.data:
                writer.writerow(row)

