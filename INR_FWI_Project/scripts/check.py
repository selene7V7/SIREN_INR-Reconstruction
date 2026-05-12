duplicates = df[df.duplicated(subset=['层数', '每层神经元数量'], keep=False)]
print(duplicates)
