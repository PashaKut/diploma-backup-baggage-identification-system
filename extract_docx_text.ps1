Add-Type -AssemblyName System.IO.Compression.FileSystem

$path = Join-Path $PSScriptRoot 'system_description.docx'
$zip = [System.IO.Compression.ZipFile]::OpenRead($path)
$entry = $zip.Entries | Where-Object { $_.FullName -eq 'word/document.xml' }
$reader = New-Object System.IO.StreamReader($entry.Open())
$content = $reader.ReadToEnd()
$reader.Close()
$zip.Dispose()

[xml]$xml = $content
$ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
$ns.AddNamespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')

$paras = $xml.SelectNodes('//w:body/w:p', $ns)
foreach ($p in $paras) {
    $texts = $p.SelectNodes('.//w:t', $ns) | ForEach-Object { $_.'#text' }
    if ($texts.Count -gt 0) {
        $texts -join ''
    }
}
