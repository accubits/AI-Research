# Lipsync-Video-Generator

Generates a video with facial movements from an input of source image and a driving video. 
Sasi Tharoor and Donald Trumps images are given in the `input_images` folder. 

### Dependencies

required packages: 

- Python 3.6

- requests==2.24
- requests-file==1.5.1
- idna==2.10
- selenium==3.141.0
- regex==2020.7.14
- sumy==0.8.1
- numpy==1.19.1
- newspaper3k==0.2.8

run `pip install -r requirements.txt` to install the required dependencies.

### Generating the summary

- Run the command `python scrape_summarize.py --search 'SEARCH TERM'` directory to generate the summary. The results saved in the `outputs` directory.
- Two other paramets are:
- * `--num_links` to select the number of links to srape from. The default value is set to `10`.
- * `--sent_count` to limit the output sentence count. The default value is set to `20`.


### Sample Outputs

```
Search term: bitcoin
Date Created: 04/09/2020 11:09:05
Links failed: 4

 Bitcoin prices tumbled 6.2% Thursday, falling below $11,000 for the first time in a month. Yet those price drops followed gains of 54% in July and 25% in August amid reports of eye-popping dollar amounts flowing into DeFi – especially with recently-launched projects like Compound, Yearn.Finance and SushiSwap – attracting attention from traders to the fast-growing and lucrative-but-risky pursuit of “yield farming.” Total value locked in DeFi more than doubled in August to $9.5 billion, but in the past few days the amount has shrunk to $9.1 billion, according to the website DeFi Pulse. Bitcoin miners and possibly traders decided to take risk off the table by trading in some of their cryptocurrency, which they receive as rewards for helping to maintain the security of the blockchain network. CoinDesk reported prior to Thursday’s sell-off that blockchain data were showing elevated transfers of bitcoin to exchange wallets, typically seen as a precursor of heightened selling pressure. Assembled by the CoinDesk Markets Team and edited by Bradley Keoun, First Mover starts your day with the most up-to-date sentiment around crypto markets, which of course never close, putting in context every wild swing in bitcoin and more. Bitcoin was down early Thursday to about $11,250, extending Wednesday’s sell-off and falling to its lowest price since early August. Cameron and Tyler Winklevoss, who run the cryptocurrency exchange Gemini, wrote last week that bitcoin prices could reach $500,000, in an extensive analysis that somehow relates to a database of 600,000 asteroids. According to CoinDesk Research’s monthly review published this week, bitcoin’s price appears to be rising whenever the dollar falls in foreign-exchange markets. Philip Bonello, director of research for the money manager Grayscale (owned by CoinDesk parent Digital Currency Group), says his favorite chart for thinking about bitcoin’s price trajectory might be one showing “holders” versus “speculators.” A holder in this case is defined as a bitcoin that has not moved for one to three years, while a speculator coin has moved in the past 90 days. “It’s reassuring,” Bonello said Wednesday in a phone interview, “that the sentiment of the investor base is growing day by day.” The holders appear to have been unfazed by the volatility witnessed in March, when the spread of the coronavirus quickly sent bitcoin prices swooning from above $9,000 to below $5,000. All of this might mean nothing for the future price of bitcoin. Bitcoin may extend Wednesday's price pullback, as exchange flows indicate increased selling pressure in the market. CoinDesk is an independent operating subsidiary of Digital Currency Group, which invests in cryptocurrencies and blockchain startups.The price of Bitcoin (BTC), the S&P 500 index, and gold all fell simultaneously on Sept. 3. What’s next for Bitcoin price? There is a substantial amount of capital on the sidelines, especially in the stablecoin market.With the price of bitcoin pulling back sharply and U.S. legislators again advocating for blockchain, CoinDesk’s Markets Daily is back for your latest crypto news roundup! Citizens and companies in Zug will be able to pay up to $109,000 of their tax bill in either bitcoin or ether as of next February.Bitcoin continues to slide while ether has a larger share of the crypto market than it has had in years. Bitcoin continues its downward trend Thursday, with prices descending as low as $10,468 on spot exchanges such as Coinbase. Just like Wednesday, leveraged liquidations played a role in exacerbating bitcoin’s price drop. Thomas suspects bitcoin’s price will not reach new 2020 highs in the near term, despite testing that level as recently as Tuesday when the price hit $12,085. But while the price is down, ether’s dominance of the broader crypto market hit a 2020 high of over 14% Wednesday.
```