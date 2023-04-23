# OpenBB Walkthroughs
## Commands vs menus
Menus are a collection of 'commands' and 'sub-menus'.
You can identify them through their distinct color and a '>' at the beginning of the line

For instance:
```bash
>   stocks             access historical pricing data, options, sector and industry, and overall due diligence 
```

Commands are expected to return data either as a chart or table.
You can identify them through their distinct color

For instance:
```bash
>   news  
```   
## Using commands
Commands throughout the terminal can have additional arguments.

Let's say that in the current menu, you want to have more information about the command `news`.

You can either see the available arguments in the terminal, using: `news -h`. I
Or you can find out more about it with an output example on the browser, using: `about news`

## Setting API Keys
The OpenBB Terminal does not own any of the data you have access to. Instead, we provide the infrastructure to access over 100 different data sources from a single location. Thus, it is necessary for each user to set their own API keys for the various third party sources

You can find more about this on the 'keys' menu.

For many commands, there are multiple data sources that can be selected.

The help menu shows the data sources supported by each command.

For instance:
```bash
load #load a specific stock ticker and additional info for analysis   [YahooFinance, AlphaVantage, Polygon, EODHD] 
```

The user can go into the `sources` menu and select their preferred default data source.

## Symbol dependent menus and commands

Throughout the terminal, you will see commands and menus greyed out.

These menus or commands cannot be accessed until an object is loaded.

Let's take as an example the 'stocks' menu.

You will see that the command 'disc' is available as its goal is to discover new tickers:
>   stocks             access historical pricing data, options, sector 

On the other hand, 'fa' menu (fundamental analysis) requires a ticker to be loaded.

And therefore, appears as:
>   fa                 fundamental analysis of loaded ticker 

Once a ticker is loaded with: load TSLA

The 'fa' menu will be available as:
>   fa                 fundamental analysis of loaded ticker 

## Terminal Navigation
The terminal has a tree like structure, where menus branch off into new menus.

The users current location is displayed before the text prompt.

For instance, if the user is inside the menu disc which is inside stocks, the following prompt will appear: 
2022 Oct 18, 21:53 (🦋) /stocks/disc/ $

If the user wants to go back to the menu above, all they need to do is type 'q'.

If the user wants to go back to the home of the terminal, they can type '/' instead.

Note: Always type 'h' to know what commands are available in each menu

## Command pipeline

The terminal offers the capability of allowing users to speed up their navigation and command execution.

Therefore, typing the following prompt is valid:
2022 Oct 18, 21:53 (🦋) / $ stocks/load TSLA/dd/pt

In this example, the terminal - in a single action - will go into 'stocks' menu, run command 'load' with 'TSLA' as input, 
go into sub-menu 'dd' (due diligence) and run the command 'pt' (price target).

## OpenBB Scripts
The command pipeline capability is great, but the user experience wasn't great copy-pasting large lists of commands.

We allow the user to create a text file of the form:
```bash	
FOLDER_PATH/my_script.openbb
stocks
load TSLA
dd
pt
```

which can be run through the 'exe' command in the home menu, with:
2022 Oct 18, 22:33 (🦋) / $ exe FOLDER_PATH/my_script.openbb

## OpenBB Scripts with arguments
The user can create a script that includes arguments for the commands.

Example:

FOLDER_PATH/my_script_with_variable_input.openbb
stocks
```bash
# this is a comment
load $ARGV[0]
dd
pt
q
load $ARGV[1]
candle
```

and then, if this script is run with:
2022 Oct 18, 22:33 (🦋) / $ `exe FOLDER_PATH/my_script_with_variable_input.openbb -i AAPL,MSFT`

This means that the pt will run on AAPL while candle on MSFT

## OpenBB Script Generation/purple

To make it easier for users to create scripts, we have created a command that 'records' user commands directly into a script.

From the home menu, the user can run:
2022 Oct 18, 22:33 (🦋) / $ record

and then perform your typical investment research workflow before entering

2022 Oct 18, 22:33 (🦋) / $ stop

After stopping, the script will be saved to the 'scripts' folder.

## Terminal Customization
Users should explore the settings and featflags menus to configure their terminal.

The fact that our terminal is fully open source allows users to be able to customize anything they want.

If you are interested in contributing to the project, please check:
https://github.com/OpenBB-finance/OpenBBTerminal

## OpenBB Terminal
### Information, guides and support for the OpenBB Terminal:
```bash
 intro              introduction on the OpenBB Terminal 
 about              discover the capabilities of the OpenBB Terminal (https://openbb.co/docs)
 support            pre-populate a support ticket for our team to evaluate                   
 survey             fill in our 2-minute survey so we better understand how we can improve the terminal│
 update             attempt to update the terminal automatically (GitHub version)            
 wiki               search for an expression in Wikipedia (https://www.wikipedia.org/)       
 news               display news articles based on term and data sources                     
```
### Configure your own terminal:
```bash
 >   keys               # set API keys and check their validity                                    
 >   featflags          # enable and disable feature flags                                         
 >   sources            # select your preferred data sources                                       
 >   settings           # tune settings (export folder, timezone, language, plot size)             
```
### Record and execute your own .openbb routine scripts:
```bash
 record             # start recording current session                                          
 stop               # stop session recording and convert to .openbb routine                    
 exe                # execute .openbb routine scripts (use exe --example for an example)
```
                                         
### Main menu:
```bash
 >   stocks             # access historical pricing data, options, sector and industry, and overall due diligence
 >   crypto             # dive into onchain data, tokenomics, circulation supply, nfts and more    
 >   etf                # exchange traded funds. Historical pricing, compare holdings and screening
 >   economy            # global macroeconomic data, e.g. futures, yield, treasury                 
 >   forex              # foreign exchanges, quotes, forward rates for currency pairs and oanda integration
 >   futures            # commodities, bonds, index, bitcoin and forex                             
 >   fixedincome        # access central bank decisions, yield curves, government bonds and corporate bonds data 
 >   alternative        # alternative datasets, such as COVID and open source metrics              
 >   funds              # mutual funds search, overview, holdings and sector weights               
```
### OpenBB Toolkits:    
```bash                
 >   econometrics       # statistical and quantitative methods for relationships between datasets  
 >   forecast           # timeseries forecasting with machine learning                             
 >   portfolio          # perform portfolio optimization and look at portfolio performance and attribution 
 >   dashboards         # interactive dashboards using voila and jupyter notebooks                 
 >   reports            # customizable research reports through jupyter notebooks

```