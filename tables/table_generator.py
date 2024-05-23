import tabula
pdf = '/home/dikshant/BOSCH/Round1/tata.pdf'

ps = tabula.read_pdf(pdf, pages='all')

for i in range(len(ps)):
    ps[i].to_csv(f'page_{i}.csv', index=False)