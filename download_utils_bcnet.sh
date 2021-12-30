# A simple bash script to download the utils from all of the related projects of Nexar's "Reconocimiento de Talla' Project.



# BCNet - From the Github repository of the project, follow the instructions to get the OneDrive link, with this instructions you should be able to get the cURL from Chrome/Firefox : https://stackoverflow.com/questions/62634214/how-to-download-protected-files-from-onedrive-using-wget
MODELSDIR=BCNet/models
NETFILE=garNet.pth


if [[ -f "$MODELSDIR/$NETFILE" ]]
then
	echo "[MSG] $NETFILE already present!"
else
	echo "[MSG] The garNet it's not present!"
	if [[	-d "$MODELSDIR"	]]
	then
		echo "[MSG] Folder <models> already exists!"
	else
		echo "[MSG] Generating <models> folder"
		mkdir $MODELSDIR
	fi
	# cURL from the onedrive
	curl 'https://mailustceducn-my.sharepoint.com/personal/jby1993_mail_ustc_edu_cn/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fjby1993%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FBCNetModel%2FgarNet%2Epth' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://mailustceducn-my.sharepoint.com/personal/jby1993_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjby1993%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FBCNetModel%2FgarNet%2Epth&parent=%2Fpersonal%2Fjby1993%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FBCNetModel' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjExLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uIzJjMzY0YjkwNDU0ZDM3MWNmNDk1YWUwZGJhMzAwZTYwMTlkOTE3MmM4YzdjNDNhMWI0ZWY1M2E5ODkxMGIxZTMsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jMmMzNjRiOTA0NTRkMzcxY2Y0OTVhZTBkYmEzMDBlNjAxOWQ5MTcyYzhjN2M0M2ExYjRlZjUzYTk4OTEwYjFlMywxMzI4NTAyODMyNTAwMDAwMDAsMCwxMzI4NTExNDQyNjg4Mzg3NTcsMC4wLjAuMCwyNTgsNWE5M2Y3YTgtODYzZi00YWViLTkwYTAtMTdkNmEwOWU3YTM1LCwsNmNhZDEwYTAtNDA0YS0zMDAwLTdiMDMtYjgwZWY3MTdkY2RiLDZjYWQxMGEwLTQwNGEtMzAwMC03YjAzLWI4MGVmNzE3ZGNkYiw0RmtDeXJLNUdFYURNTnJ4Qk1KUnpRLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLEtxQkxaWWIwd0YvSWU1K1FlenMrencvc05IRGoyYXZzQ3I2emJGUFhNVjlscXl4WnpxT0tzQnNGcE1XeTJVbUhmMm5leE1FOVBwYUhsVEc3VG01UWNOVEgzb0loZm5ac2d3MlhFaWdYUndkK3RoTEh4U3FQT1E2cTJTN3hDS3l3TDgrMzBIVDdMcTFpOXEvcGJMRjNxMWxGL0RNTXVFayswWkZpU2dzMGhLQldIeVBGN3JOelNxWXhMcmNPOTBWM3hzczUxYk5ZMjRtT3Q4L1dlSXVUN3FBMWVCWStzR1p2TU9uVUhScCtpamd1VzlHZ05ZMjZzaCtMNjlTZnZBWUlEYVAzOHp2S2pGZG10L1pBbHJHVmcvL1Blem5XWUdXTnVTay9nNGFLVndVODA2V3FBSWN6SU5ZZnpLazY3aFZPSWZhSitlbkZxYWMrMTZ6YjZlMnhwZz09PC9TUD4=; KillSwitchOverrides_enableKillSwitches=; KillSwitchOverrides_disableKillSwitches=' --output $MODELSDIR/$NETFILE

	echo "[MSG] garNet downloaded!"
fi

# From the Github repository get the link for the tmps.rar file from OneDrive:

TMPS_FILE=tmps.rar
BODY_PATH=BCNet/body_garment_dataset

if [[ -f "$BODY_PATH/$TMPS_FILE" ]]
then
	echo "[MSG] $TMPS_FILE already present!"
else
	echo "[MSG] The $TMPS_FILE it's not present!"
	curl 'https://mailustceducn-my.sharepoint.com/personal/jby1993_mail_ustc_edu_cn/_layouts/15/download.aspx?UniqueId=f9156614%2Dbb1e%2D4278%2Da88e%2D03ad486ced2e' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://mailustceducn-my.sharepoint.com/personal/jby1993_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjby1993%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fbody%5Fgarment%5Fdataset' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjExLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2RhZDg5YmE3NGI4Yjk0ZmVjZTViNWQ1YjA5N2U4OWZhYTViYTM2OGY0NWM5ZDI2MDE4ODg4MjBjOWY3ODY2YWIsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jZGFkODliYTc0YjhiOTRmZWNlNWI1ZDViMDk3ZTg5ZmFhNWJhMzY4ZjQ1YzlkMjYwMTg4ODgyMGM5Zjc4NjZhYiwxMzI4NTA0NDk1MDAwMDAwMDAsMCwxMzI4NTEzMTA1MDY3NjgyOTIsMC4wLjAuMCwyNTgsNWE5M2Y3YTgtODYzZi00YWViLTkwYTAtMTdkNmEwOWU3YTM1LCwsNDdiZDEwYTAtMTAwYS0zMDAwLTdiMDMtYjJmNDU5N2YwN2RkLDQ3YmQxMGEwLTEwMGEtMzAwMC03YjAzLWIyZjQ1OTdmMDdkZCxsbnR2cTVkZDJrS3NJVVZSeWdiaUlnLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLE0xdlNjWDJNWUhEL1VJbDZzcDIreWpueXpudW0rNUR1VVBlM3pOOWlLZnROOUpEaW0xanNwU0VEWU5IZ2RaSFMwQlpiQTM0VXdNTHpQYUJqVk9WZWlXWnVvQXBydE9FVEg4b0JkckUvOTROUTVyblVhRUUzWWU0WVVGeGpDRXdhbkhraHJHMDVyMDdnVmhzT1REb3dyUUttZE9Xdi9PTkZqbldZWVdQYnZxdHc5TVpUcDRyUnZaVXN5K1RhWHd6ajdPU2FwallMN2RuK0xTajVCU0JHdndVenpBclVBVXRwd3J1eTgzY0tLRmZRTVBGMjA2dmpyR0toL0pmaFJGblRHMEE4Q3c1OC9RMEE5WUc4QjkvVXlFVjZVMGh1K2s4VWk0ZElVVGwwdVY0anNJTDZSRG1BcU9NQzFodmtQSzBWM3lEaWhSRmMyN1RncDVId2FTcWlUdz09PC9TUD4=; KillSwitchOverrides_enableKillSwitches=; KillSwitchOverrides_disableKillSwitches=' --output $BODY_PATH/$TMPS_FILE
	unrar x $BODY_PATH/$TMPS_FILE $BODY_PATH/
fi


# From the smpl_pytorch folder we can get the link for the neutral_smpl_with_cocoplus_reg.txt
SMPL_PYTORCH_PATH=BCNet/smpl_pytorch/model
SMPL_PYTORCH_FILE=neutral_smpl_with_cocoplus_reg.txt


if [[	-f  $SMPL_PYTORCH_PATH/$SMPL_PYTORCH_FILE	]]
then
	echo "[MSG] $SMPL_PYTORCH_FILE already present!"
else
	mkdir $SMPL_PYTORCH_PATH
	curl 'https://mailustceducn-my.sharepoint.com/personal/jby1993_mail_ustc_edu_cn/_layouts/15/download.aspx?UniqueId=dd5fee68%2D3d70%2D4fd9%2D84c8%2D9bddab3b36b3' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://mailustceducn-my.sharepoint.com/personal/jby1993_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjby1993%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fmodel' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjExLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2RhZDg5YmE3NGI4Yjk0ZmVjZTViNWQ1YjA5N2U4OWZhYTViYTM2OGY0NWM5ZDI2MDE4ODg4MjBjOWY3ODY2YWIsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jZGFkODliYTc0YjhiOTRmZWNlNWI1ZDViMDk3ZTg5ZmFhNWJhMzY4ZjQ1YzlkMjYwMTg4ODgyMGM5Zjc4NjZhYiwxMzI4NTA0NDk1MDAwMDAwMDAsMCwxMzI4NTEzMTA1MDY3NjgyOTIsMC4wLjAuMCwyNTgsNWE5M2Y3YTgtODYzZi00YWViLTkwYTAtMTdkNmEwOWU3YTM1LCwsNDdiZDEwYTAtMTAwYS0zMDAwLTdiMDMtYjJmNDU5N2YwN2RkLDQ3YmQxMGEwLTEwMGEtMzAwMC03YjAzLWIyZjQ1OTdmMDdkZCxsbnR2cTVkZDJrS3NJVVZSeWdiaUlnLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLE0xdlNjWDJNWUhEL1VJbDZzcDIreWpueXpudW0rNUR1VVBlM3pOOWlLZnROOUpEaW0xanNwU0VEWU5IZ2RaSFMwQlpiQTM0VXdNTHpQYUJqVk9WZWlXWnVvQXBydE9FVEg4b0JkckUvOTROUTVyblVhRUUzWWU0WVVGeGpDRXdhbkhraHJHMDVyMDdnVmhzT1REb3dyUUttZE9Xdi9PTkZqbldZWVdQYnZxdHc5TVpUcDRyUnZaVXN5K1RhWHd6ajdPU2FwallMN2RuK0xTajVCU0JHdndVenpBclVBVXRwd3J1eTgzY0tLRmZRTVBGMjA2dmpyR0toL0pmaFJGblRHMEE4Q3c1OC9RMEE5WUc4QjkvVXlFVjZVMGh1K2s4VWk0ZElVVGwwdVY0anNJTDZSRG1BcU9NQzFodmtQSzBWM3lEaWhSRmMyN1RncDVId2FTcWlUdz09PC9TUD4=; KillSwitchOverrides_enableKillSwitches=; KillSwitchOverrides_disableKillSwitches=' --output $SMPL_PYTORCH_PATH/$SMPL_PYTORCH_FILE
fi
