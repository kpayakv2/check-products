$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.IO.Path]::Combine($env:USERPROFILE, "Desktop")
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\PHAYAK_AI.lnk")
$Shortcut.TargetPath = "D:\product_checker\check-products\START_PHAYAK.bat"
$Shortcut.WorkingDirectory = "D:\product_checker\check-products"
$Shortcut.IconLocation = "imageres.dll, 164"
$Shortcut.Description = "Neural Taxonomy Engine - Phayak AI"
$Shortcut.Save()
Write-Host "✅ PHAYAK AI Shortcut has been placed on your Desktop!"
