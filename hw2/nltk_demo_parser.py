#http://www.nltk.org/getting-started
#sudo python -m nltk.downloader -d /usr/share/nltk_data all
import nltk

#http://nltk.googlecode.com/svn/trunk/doc/api/nltk.parse.earleychart.EarleyChartParser-class.html

grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  NP -> Adj NP | PRP | N
  VP -> V NP | Aux V NP
  PRP -> "they"
  N -> "potatoes"
  Adj -> "baking"
  V -> "baking" | "are"
  Aux -> "are"
  """)

sent = "they are baking potatoes".split()

parser = nltk.parse.EarleyChartParser(grammar1)
chart = parser.chart_parse(sent, trace=1)
