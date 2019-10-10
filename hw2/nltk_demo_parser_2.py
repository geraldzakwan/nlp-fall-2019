#http://www.nltk.org/getting-started
#sudo python -m nltk.downloader -d /usr/share/nltk_data all
import nltk

#http://nltk.googlecode.com/svn/trunk/doc/api/nltk.parse.earleychart.EarleyChartParser-class.html

grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)

sent = "Mary saw Bob".split()

parser = nltk.parse.EarleyChartParser(grammar1)
chart = parser.chart_parse(sent, trace=1)
