/* build_doc.js — generate .docx from markdown using docx library */
const fs=require('fs'),{Document,Packer,Paragraph,TextRun,HeadingLevel,Table,TableRow,TableCell,WidthType}=require('docx');

async function main(){const doc=new Document({sections:[{children:[new Paragraph({text:'shuanzhi Technical Report',heading:HeadingLevel.TITLE})]}]});
    const buf=await Packer.toBuffer(doc);fs.writeFileSync('output.docx',buf);console.log('Done');}
main().catch(console.error);
