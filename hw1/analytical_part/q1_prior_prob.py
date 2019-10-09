from collections import defaultdict

d1 = ('spam', ('buy', 'car', 'Nigeria', 'profit', ), )
d2 = ('spam', ('money', 'profit', 'home', 'bank', ), )
d3 = ('spam', ('Nigeria', 'bank', 'check', 'wire', ), )
d4 = ('ham', ('money', 'bank', 'home', 'car', ), )
d5 = ('ham', ('home', 'Nigeria', 'fly', ), )

data = (d1, d2, d3, d4, d5, )

# count(Word, Class)
count_word_given_class_dict = defaultdict(int)

# count(All Word, Class)
count_all_words_given_class_dict = defaultdict(int)

for email in data:
    label = email[0]

    for word in email[1]:
        count_word_given_class_dict[(word, label)] += 1
        count_all_words_given_class_dict[label] += 1

# for key in count_word_given_class_dict:
#     if key[1] == 'spam':
#         print(str(key) + ': ' + str(count_word_given_class_dict[key]) + '/' + str(count_all_words_given_class_dict[key[1]]))
#         print()
#
# print('-------')
# print('-------')
#
# for key in count_word_given_class_dict:
#     if key[1] == 'ham':
#         print(str(key) + ': ' + str(count_word_given_class_dict[key]) + '/' + str(count_all_words_given_class_dict[key[1]]))
#         print()
#
#
# words = 0
# for email in data:
#     words += len(email[1])
#
# print('Total words: ')
# print(words)

while True:
    x = input()
    y = input()

    print(str(count_word_given_class_dict[(x,y)]) + '/' + str(count_all_words_given_class_dict[y]))
